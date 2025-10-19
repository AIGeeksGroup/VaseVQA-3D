#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import numpy as np
import torch
from PIL import Image
import trimesh
import random
import gc
import glob
import time
import threading
import queue
import multiprocessing
import psutil
import signal
import subprocess
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError

# 设置环境变量来限制并行处理
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512,expandable_segments:True"
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16

print(f"DEVICE: {DEVICE}")
print(f"CUDA Available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

DEFAULT_FACE_NUMBER = 50000
RMBG_PRETRAINED_MODEL = "checkpoints/RMBG-1.4"
TRIPOSG_PRETRAINED_MODEL = "checkpoints/TripoSG"

TMP_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output2")
os.makedirs(TMP_DIR, exist_ok=True)

TRIPOSG_CODE_DIR = "./triposg"
MV_ADAPTER_CODE_DIR = "./mv_adapter"

# 添加路径
sys.path.append(TRIPOSG_CODE_DIR)
sys.path.append(os.path.join(TRIPOSG_CODE_DIR, "scripts"))
sys.path.append(MV_ADAPTER_CODE_DIR)
sys.path.append(os.path.join(MV_ADAPTER_CODE_DIR, "scripts"))

print("正在导入模块...")

try:
    from triposg.pipelines.pipeline_triposg import TripoSGPipeline
    print("✅ TripoSG pipeline 导入成功")
except Exception as e:
    print(f"❌ TripoSG pipeline 导入失败: {e}")
    sys.exit(1)

try:
    from mv_adapter.mvadapter.utils import get_orthogonal_camera, tensor_to_image, make_image_grid
    from mv_adapter.mvadapter.utils.render import NVDiffRastContextWrapper, load_mesh, render
    print("✅ MV-Adapter utils 导入成功")
except Exception as e:
    print(f"❌ MV-Adapter utils 导入失败: {e}")
    sys.exit(1)

try:
    from texture import TexturePipeline, ModProcessConfig
    print("✅ Texture pipeline 导入成功")
except Exception as e:
    print(f"❌ Texture pipeline 导入失败: {e}")
    sys.exit(1)

try:
    from image_process import prepare_image
    print("✅ Image process 导入成功")
except Exception as e:
    print(f"❌ Image process 导入失败: {e}")
    sys.exit(1)

try:
    from inference_ig2mv_sdxl import prepare_pipeline, preprocess_image, remove_bg
    print("✅ Inference pipeline 导入成功")
except Exception as e:
    print(f"❌ Inference pipeline 导入失败: {e}")
    sys.exit(1)

try:
    from briarmbg import BriaRMBG
    print("✅ BriaRMBG 导入成功")
except Exception as e:
    print(f"❌ BriaRMBG 导入失败: {e}")
    sys.exit(1)
    
try:
    from transformers import AutoModelForImageSegmentation
    from torchvision import transforms
    print("✅ Transformers 导入成功")
except Exception as e:
    print(f"❌ Transformers 导入失败: {e}")
    sys.exit(1)

class ThreadTimeoutError(Exception):
    """线程超时异常"""
    pass

class ProcessKiller:
    """进程终止器，用于强制终止子进程"""
    def __init__(self):
        self.processes = []
    
    def add_process(self, process):
        self.processes.append(process)
    
    def kill_all(self):
        """强制终止所有进程"""
        for proc in self.processes:
            try:
                if proc.poll() is None:  # 进程还在运行
                    proc.terminate()
                    proc.wait(timeout=5)
            except:
                try:
                    proc.kill()
                except:
                    pass
        self.processes.clear()

def run_with_thread_timeout(func, args=(), kwargs=None, timeout_seconds=120):
    """
    在线程中运行函数，如果超时则强制退出
    """
    if kwargs is None:
        kwargs = {}
    
    result_queue = queue.Queue()
    exception_queue = queue.Queue()
    
    def target():
        try:
            result = func(*args, **kwargs)
            result_queue.put(result)
        except Exception as e:
            exception_queue.put(e)
    
    thread = threading.Thread(target=target)
    thread.daemon = True  # 设置为守护线程
    thread.start()
    
    # 等待结果或超时
    thread.join(timeout=timeout_seconds)
    
    if thread.is_alive():
        # 线程仍在运行，说明超时了
        print(f"⏰ 操作超时 ({timeout_seconds}秒)，强制终止...")
        # 注意：Python中无法直接终止线程，但守护线程会在主程序退出时自动终止
        raise ThreadTimeoutError(f"操作超时 ({timeout_seconds}秒)")
    
    # 检查是否有异常
    if not exception_queue.empty():
        raise exception_queue.get()
    
    # 检查是否有结果
    if not result_queue.empty():
        return result_queue.get()
    
    raise ThreadTimeoutError("操作未返回结果")

def run_texture_with_subprocess(mesh_path, save_dir, save_name, uv_size, rgb_path, base_name, timeout_seconds=120):
    """
    使用子进程运行纹理处理，可以强制终止
    """
    # 创建一个临时的Python脚本来执行纹理处理
    temp_script = os.path.join(TMP_DIR, f"temp_texture_{base_name}.py")
    
    script_content = f'''
import sys
import os
sys.path.append("./triposg")
sys.path.append("./mv_adapter")
import torch
from texture import TexturePipeline, ModProcessConfig

try:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    texture_pipe = TexturePipeline(
        upscaler_ckpt_path="checkpoints/RealESRGAN_x2plus.pth",
        inpaint_ckpt_path="checkpoints/big-lama.pt",
        device=DEVICE,
    )
    
    result = texture_pipe(
        mesh_path="{mesh_path}",
        save_dir="{save_dir}",
        save_name="{save_name}",
        uv_unwarp=True,
        uv_size={uv_size},
        rgb_path="{rgb_path}",
        rgb_process_config=ModProcessConfig(
            view_upscale={uv_size >= 2048},
            inpaint_mode="view"
        ),
        camera_azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
    )
    
    print(f"SUCCESS:{{result.shaded_model_save_path if hasattr(result, 'shaded_model_save_path') else str(result)}}")
    
except Exception as e:
    print(f"ERROR:{{str(e)}}")
    import traceback
    traceback.print_exc()
'''
    
    # 写入临时脚本
    with open(temp_script, 'w', encoding='utf-8') as f:
        f.write(script_content)
    
    try:
        # 启动子进程
        process = subprocess.Popen(
            [sys.executable, temp_script],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            preexec_fn=os.setsid  # 创建新的进程组
        )
        
        # 等待进程完成或超时
        try:
            stdout, stderr = process.communicate(timeout=timeout_seconds)
            
            # 解析输出
            if "SUCCESS:" in stdout:
                result_path = stdout.split("SUCCESS:")[1].strip()
                return result_path
            elif "ERROR:" in stdout:
                error_msg = stdout.split("ERROR:")[1].strip()
                raise Exception(f"子进程错误: {error_msg}")
            else:
                raise Exception(f"子进程未返回预期结果: {stdout}")
                
        except subprocess.TimeoutExpired:
            print(f"⏰ 纹理处理超时 ({timeout_seconds}秒)，强制终止进程...")
            
            # 强制终止进程组
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
                time.sleep(2)
                if process.poll() is None:
                    os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            except:
                pass
            
            process.kill()
            process.wait()
            raise ThreadTimeoutError(f"纹理处理超时 ({timeout_seconds}秒)")
            
    finally:
        # 清理临时脚本
        try:
            os.remove(temp_script)
        except:
            pass

def force_cleanup_memory():
    """强制清理GPU和CPU内存"""
    try:
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except:
        pass
    gc.collect()
    
    # 打印当前内存使用情况
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"  💾 GPU内存: 已分配 {allocated:.2f}GB, 已缓存 {cached:.2f}GB")

def repair_mesh_for_uv(mesh):
    """强力修复网格，确保可以进行UV展开"""
    print("    正在强力修复网格以支持UV展开...")
    try:
        # 步骤1: 基础清理
        mesh.update_faces(mesh.unique_faces())
        
        # 步骤2: 修复法线
        mesh.fix_normals()
        
        # 步骤3: 填充小孔洞
        mesh.fill_holes()
        
        # 步骤4: 分离连通组件，保留最大的
        components = mesh.split(only_watertight=False)
        if len(components) > 1:
            print(f"      发现{len(components)}个连通组件，保留最大的")
            mesh = max(components, key=lambda x: len(x.vertices))
        
        # 步骤5: 强制使网格成为流形
        try:
            if not mesh.is_watertight:
                mesh.remove_unreferenced_vertices()
                
                if not mesh.is_watertight:
                    convex_mesh = mesh.convex_hull
                    if convex_mesh.is_watertight:
                        mesh = convex_mesh
        except Exception as e:
            print(f"      流形化过程中出现警告: {e}")
        
        print(f"      网格修复完成 - 顶点: {len(mesh.vertices)}, 面: {len(mesh.faces)}, 水密: {mesh.is_watertight}")
        return mesh
        
    except Exception as e:
        print(f"      网格修复失败: {e}")
        return mesh

class TripoSGBatchProcessor:
    def __init__(self):
        self.rmbg_net = None
        self.triposg_pipe = None
        self.mv_adapter_pipe = None
        self.remove_bg_fn = None
        self.texture_pipe = None
        self.NUM_VIEWS = 6
        self.process_killer = ProcessKiller()
        
    def setup_models(self):
        """一次性初始化所有模型"""
        print("正在初始化所有模型...")
        
        # 初始化前先清理内存
        force_cleanup_memory()
        
        # RMBG模型
        print("  正在初始化RMBG模型...")
        if os.path.exists(RMBG_PRETRAINED_MODEL):
            self.rmbg_net = BriaRMBG.from_pretrained(RMBG_PRETRAINED_MODEL).to(DEVICE)
            self.rmbg_net.eval()
            print("  ✅ RMBG模型初始化完成")
        else:
            print(f"  ❌ RMBG模型文件夹不存在: {RMBG_PRETRAINED_MODEL}")
            return False
        
        # TripoSG模型
        print("  正在初始化TripoSG模型...")
        if os.path.exists(TRIPOSG_PRETRAINED_MODEL):
            self.triposg_pipe = TripoSGPipeline.from_pretrained(TRIPOSG_PRETRAINED_MODEL).to(DEVICE, DTYPE)
            print("  ✅ TripoSG模型初始化完成")
        else:
            print(f"  ❌ TripoSG模型文件夹不存在: {TRIPOSG_PRETRAINED_MODEL}")
            return False
        
        # MV-Adapter
        print("  正在初始化MV-Adapter...")
        os.environ["HF_HUB_OFFLINE"] = "1"
        self.mv_adapter_pipe = prepare_pipeline(
            base_model="stabilityai/stable-diffusion-xl-base-1.0",
            vae_model="madebyollin/sdxl-vae-fp16-fix",
            unet_model=None,
            lora_model=None,
            adapter_path="huanngzh/mv-adapter",
            scheduler=None,
            num_views=self.NUM_VIEWS,
            device=DEVICE,
            dtype=torch.float16,
        )
        print("  ✅ MV-Adapter初始化完成")
        
        # BiRefNet
        print("  正在初始化BiRefNet...")
        birefnet = AutoModelForImageSegmentation.from_pretrained(
            "ZhengPeng7/BiRefNet", trust_remote_code=True
        )
        birefnet.to(DEVICE)
        transform_image = transforms.Compose([
            transforms.Resize((1024, 1024)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        self.remove_bg_fn = lambda x: remove_bg(x, birefnet, transform_image, DEVICE)
        print("  ✅ BiRefNet初始化完成")
        
        # 纹理管道初始化推迟到使用时（因为容易卡死）
        print("  📝 纹理管道将在使用时初始化")
        
        print("🎉 模型初始化完成！")
        return True
    
    def generate_3d_mesh(self, image_seg, seed=0):
        """生成3D网格（带超时控制）"""
        def _generate():
            num_inference_steps = 20
            guidance_scale = 7.5
            
            outputs = self.triposg_pipe(
                image=image_seg,
                generator=torch.Generator(device=self.triposg_pipe.device).manual_seed(seed),
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale
            ).samples[0]
            
            # 确保outputs是正确的格式
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                vertices, faces = outputs[0], outputs[1]
                # 转换为numpy数组
                if hasattr(vertices, 'cpu'):
                    vertices = vertices.cpu().numpy()
                if hasattr(faces, 'cpu'):
                    faces = faces.cpu().numpy()
                mesh = trimesh.Trimesh(vertices.astype(np.float32), np.ascontiguousarray(faces))
                return mesh
            else:
                raise Exception("TripoSG输出格式不正确")
        
        return run_with_thread_timeout(_generate, timeout_seconds=90)
    
    def generate_multiview_images(self, mesh_path, reference_image, seed=0):
        """生成多视角图像（带超时控制）"""
        def _generate():
            height, width = 768, 768
            
            cameras = get_orthogonal_camera(
                elevation_deg=[0, 0, 0, 0, 89.99, -89.99],
                distance=[1.8] * self.NUM_VIEWS,
                left=-0.55,
                right=0.55,
                bottom=-0.55,
                top=0.55,
                azimuth_deg=[x - 90 for x in [0, 90, 180, 270, 180, 180]],
                device=DEVICE,
            )
            
            # 渲染控制图像
            ctx = NVDiffRastContextWrapper(device=DEVICE, context_type="cuda")
            mesh_render = load_mesh(mesh_path, rescale=True, device=DEVICE)
            render_out = render(
                ctx,
                mesh_render,
                cameras,
                height=height,
                width=width,
                render_attr=False,
                normal_background=0.0,
            )
            
            if render_out is None or not hasattr(render_out, 'pos') or not hasattr(render_out, 'normal'):
                raise Exception("渲染输出无效")
            
            control_images = (
                torch.cat([
                    (render_out.pos + 0.5).clamp(0, 1),
                    (render_out.normal / 2 + 0.5).clamp(0, 1),
                ], dim=-1,)
                .permute(0, 3, 1, 2)
                .to(DEVICE)
            )
            
            # 清理渲染资源
            del render_out, mesh_render, cameras, ctx
            force_cleanup_memory()
            
            # 生成多视角图像
            pipe_kwargs = {"generator": torch.Generator(device=DEVICE).manual_seed(seed)}
            
            try:
                images = self.mv_adapter_pipe(
                    "high quality",
                    height=height,
                    width=width,
                    num_inference_steps=15,
                    guidance_scale=3.0,
                    num_images_per_prompt=self.NUM_VIEWS,
                    control_image=control_images,
                    control_conditioning_scale=1.0,
                    reference_image=reference_image,
                    reference_conditioning_scale=1.0,
                    negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
                    cross_attention_kwargs={"scale": 1.0},
                    **pipe_kwargs,
                ).images
                
                if images is None:
                    raise Exception("MV-Adapter返回了None")
                
                return images
                
            except torch.cuda.OutOfMemoryError:
                print("  内存不足，切换到分批处理模式...")
                force_cleanup_memory()
                
                # 分批处理
                images = []
                batch_size = 3
                
                for batch_start in range(0, self.NUM_VIEWS, batch_size):
                    batch_end = min(batch_start + batch_size, self.NUM_VIEWS)
                    batch_control = control_images[batch_start:batch_end]
                    
                    batch_images = self.mv_adapter_pipe(
                        "high quality",
                        height=height,
                        width=width,
                        num_inference_steps=15,
                        guidance_scale=3.0,
                        num_images_per_prompt=batch_end - batch_start,
                        control_image=batch_control,
                        control_conditioning_scale=1.0,
                        reference_image=reference_image,
                        reference_conditioning_scale=1.0,
                        negative_prompt="watermark, ugly, deformed, noisy, blurry, low contrast",
                        cross_attention_kwargs={"scale": 1.0},
                        **pipe_kwargs,
                    ).images
                    
                    if batch_images is None:
                        raise Exception("MV-Adapter批次返回了None")
                    
                    images.extend(batch_images)
                    
                    # 清理批次内存
                    del batch_control, batch_images
                    force_cleanup_memory()
                
                return images
        
        return run_with_thread_timeout(_generate, timeout_seconds=120)
    
    def apply_texture_with_timeout(self, mesh_path, base_name, mv_image_path):
        """应用纹理（带严格超时控制）"""
        print("步骤4: 正在应用纹理...")
        
        textured_path = mesh_path  # 默认值
        
        # 尝试不同的UV分辨率，使用子进程执行
        for uv_size, strategy_name, timeout_sec in [(4096, "4K", 120), (2048, "2K", 90), (1024, "1K", 60)]:
            try:
                print(f"  尝试{strategy_name}分辨率UV展开... (超时: {timeout_sec}秒)")
                
                save_name = f"{base_name}_textured_{strategy_name.lower()}.glb"
                
                # 使用子进程处理纹理
                result_path = run_texture_with_subprocess(
                    mesh_path=mesh_path,
                    save_dir=TMP_DIR,
                    save_name=save_name,
                    uv_size=uv_size,
                    rgb_path=mv_image_path,
                    base_name=base_name,
                    timeout_seconds=timeout_sec
                )
                
                print(f"  ✅ {strategy_name}纹理应用成功!")
                textured_path = result_path
                break
                
            except ThreadTimeoutError as e:
                print(f"  ⏰ {strategy_name}纹理应用超时: {e}")
                if uv_size == 1024:  # 最后一次尝试
                    print("  ❌ 所有纹理策略都超时，使用基础网格")
                    textured_path = mesh_path
            except Exception as e:
                print(f"  ❌ {strategy_name}纹理应用失败: {e}")
                if uv_size == 1024:  # 最后一次尝试
                    print("  ❌ 所有纹理策略都失败，使用基础网格")
                    textured_path = mesh_path
            finally:
                # 强制清理内存
                force_cleanup_memory()
        
        return textured_path
    
    def process_single_image(self, image_path, base_name):
        """处理单张图片"""
        print(f"\n{'='*60}")
        print(f"开始处理: {image_path}")
        print(f"输出前缀: {base_name}")
        print(f"{'='*60}")
        
        # 开始前先清理一次内存
        force_cleanup_memory()
        
        try:
            # 使用5分钟总超时
            return run_with_thread_timeout(
                self._process_single_image_internal,
                args=(image_path, base_name),
                timeout_seconds=300  # 5分钟
            )
            
        except ThreadTimeoutError:
            print(f"\n⏰ {base_name} 总处理超时 (5分钟)，强制跳过")
            # 强制终止所有子进程
            self.process_killer.kill_all()
            force_cleanup_memory()
            return {'success': False, 'error': '总处理超时 (5分钟)'}
        except Exception as e:
            print(f"\n❌ {base_name} 处理失败: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 失败时清理资源
            self.process_killer.kill_all()
            force_cleanup_memory()
            
            return {'success': False, 'error': str(e)}
    
    def _process_single_image_internal(self, image_path, base_name):
        """内部处理单张图片的实际逻辑"""
        try:
            # 步骤1: 图像分割
            print("步骤1: 正在进行图像分割...")
            image_seg = prepare_image(image_path, bg_color=np.array([1.0, 1.0, 1.0]), rmbg_net=self.rmbg_net)
            
            if image_seg is None:
                raise Exception("图像分割失败，返回了None")
            
            seg_save_path = os.path.join(TMP_DIR, f"{base_name}_segmented.png")
            image_seg.save(seg_save_path)
            print(f"  ✅ 分割完成: {seg_save_path}")
            
            # 步骤2: 生成3D网格
            print("步骤2: 正在生成3D网格...")
            mesh = self.generate_3d_mesh(image_seg)
            
            # 清理GPU内存
            force_cleanup_memory()
            
            # 修复网格
            mesh = repair_mesh_for_uv(mesh)
            
            # 简化网格
            print("  正在简化网格...")
            try:
                from utils import simplify_mesh
                mesh = simplify_mesh(mesh, DEFAULT_FACE_NUMBER)
                mesh = repair_mesh_for_uv(mesh)
            except Exception as e:
                print(f"  警告: 网格简化失败: {e}")
            
            mesh_path = os.path.join(TMP_DIR, f"{base_name}_mesh.glb")
            mesh.export(mesh_path)
            print(f"  ✅ 网格生成完成: {mesh_path}")
            
            # 步骤3: 生成多视角图像
            print("步骤3: 正在生成多视角图像...")
            
            # 准备参考图像
            image = Image.open(image_path)
            if self.remove_bg_fn is None:
                raise Exception("背景移除函数未初始化")
            image = self.remove_bg_fn(image)
            image = preprocess_image(image, 768, 768)
            
            # 生成多视角图像
            images = self.generate_multiview_images(mesh_path, image)
            
            # 清理内存
            del image
            force_cleanup_memory()
            
            mv_image_path = os.path.join(TMP_DIR, f"{base_name}_multiview.png")
            make_image_grid(images, rows=1).save(mv_image_path)
            print(f"  ✅ 多视角图像生成完成: {mv_image_path}")
            
            # 清理images
            del images
            force_cleanup_memory()
            
            # 步骤4: 应用纹理（使用强制超时控制）
            textured_path = self.apply_texture_with_timeout(mesh_path, base_name, mv_image_path)
            
            print(f"\n🎉 {base_name} 处理完成!")
            print(f"📁 分割图像: {seg_save_path}")
            print(f"📁 网格文件: {mesh_path}")
            print(f"📁 多视角图像: {mv_image_path}")
            print(f"📁 最终模型: {textured_path}")
            
            return {
                'success': True,
                'segmented': seg_save_path,
                'mesh': mesh_path,
                'multiview': mv_image_path,
                'textured': textured_path
            }
            
        except Exception as e:
            raise e
    
    def process_batch(self, input_dir):
        """批处理文件夹中的所有图片"""
        # 查找所有图片文件
        image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.PNG', '*.JPG', '*.JPEG']
        image_files = []
        
        for ext in image_extensions:
            image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        
        if not image_files:
            print(f"❌ 在 {input_dir} 中没有找到图片文件")
            return
        
        print(f"找到 {len(image_files)} 个图片文件:")
        for img in image_files:
            print(f"  - {os.path.basename(img)}")
        
        # 初始化模型
        if not self.setup_models():
            print("❌ 模型初始化失败")
            return
        
        # 处理每张图片
        results = []
        for i, image_path in enumerate(image_files, 1):
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            
            print(f"\n{'#'*80}")
            print(f"处理进度: {i}/{len(image_files)} - {base_name}")
            print(f"{'#'*80}")
            
            # 处理前强制清理内存
            force_cleanup_memory()
            
            result = self.process_single_image(image_path, base_name)
            result['image_path'] = image_path
            result['base_name'] = base_name
            results.append(result)
            
            # 处理后强制清理内存和进程
            self.process_killer.kill_all()
            force_cleanup_memory()
            
            # 每处理3张图片额外清理一次
            if i % 3 == 0:
                print(f"  📊 已处理{i}张图片，执行深度清理...")
                time.sleep(3)  # 等待3秒让所有资源完全释放
                force_cleanup_memory()
                
                # 检查并打印内存统计
                if torch.cuda.is_available():
                    max_memory = torch.cuda.max_memory_allocated() / 1024**3
                    print(f"  📈 峰值GPU内存使用: {max_memory:.2f}GB")
                    torch.cuda.reset_peak_memory_stats()
        
        # 输出总结
        print(f"\n{'='*80}")
        print("批处理完成总结:")
        print(f"{'='*80}")
        
        success_count = sum(1 for r in results if r['success'])
        print(f"成功处理: {success_count}/{len(results)}")
        
        for result in results:
            if result['success']:
                print(f"✅ {result['base_name']}")
                print(f"   分割图像: {result['segmented']}")
                print(f"   网格文件: {result['mesh']}")
                print(f"   多视角图像: {result['multiview']}")
                print(f"   最终模型: {result['textured']}")
            else:
                print(f"❌ {result['base_name']}: {result['error']}")
        
        print(f"\n🎉 批处理完成! 成功率: {success_count}/{len(results)}")

def main():
    if len(sys.argv) != 2:
        print("使用方法: python triposg_batch_threaded.py <输入目录>")
        print("示例: python triposg_batch_threaded.py test/")
        sys.exit(1)
    
    input_dir = sys.argv[1]
    if not os.path.exists(input_dir):
        print(f"错误: 找不到输入目录 {input_dir}")
        sys.exit(1)
    
    processor = TripoSGBatchProcessor()
    processor.process_batch(input_dir)

if __name__ == "__main__":
    main()