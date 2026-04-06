# main.py - 化学逆合成预测服务（基于 CBRetro）
import uuid
import time
from datetime import datetime
from typing import AsyncGenerator
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import StreamingResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import asyncio
import sys
import os
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from utils.show_smiles import reaction_smiles_to_image_base64
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from ChemBart4web import CBRetro




# ========== 1. 初始化化学模型 ==========
print("正在加载化学逆合成模型...")
MODEL_PATH = "../retrain_pv_temp/model/ChemBart_Full4.pth"  # ⚠️ 请替换为你的模型实际路径
#MODEL_PATH = "../retrain_pv_temp/model/ChemBart_MIT9.pth"
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

try:
    chem_model = CBRetro(path=MODEL_PATH, dev=DEVICE)
    print("化学模型加载完成！")
except Exception as e:
    print(f"模型加载失败: {e}")
    raise

# ========== 2. 简单内存限流器 ==========
RATE_LIMIT_STORE = {}
REQUESTS_PER_DAY = 1000  # 可根据需求调整

def check_rate_limit(user_id: str) -> bool:
    '''
    now = time.time()
    user_data = RATE_LIMIT_STORE.get(user_id, {"count": 0, "reset_time": now + 86400})

    if now > user_data["reset_time"]:
        user_data = {"count": 0, "reset_time": now + 86400}

    if user_data["count"] >= REQUESTS_PER_DAY:
        return False

    user_data["count"] += 1
    RATE_LIMIT_STORE[user_id] = user_data
    '''
    return True

# ========== 3. 创建 FastAPI 应用 ==========
app = FastAPI(title="ChemBart")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")

# ========== 4. 首页路由 ==========
@app.get("/", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    user_id = request.cookies.get("user_id")
    if not user_id:
        user_id = str(uuid.uuid4())
    response = templates.TemplateResponse("single_step_cn.html", {"request": request})
    response.set_cookie(key="user_id", value=user_id, httponly=True, max_age=86400*30)
    return response

# ========== 5. 核心逆合成预测接口（流式）==========
@app.get("/predict/reactant")
async def chat_endpoint(
    request: Request,
    q: str,
    top_k: int = 10,
    top_p: float = 0.9,
    sampling_method: str = 'beam',
    num_samples: int = 10,
    temperature: float = 1.0
):
    # 参数校验
    '''
    if sampling_method not in ['beam', 'top_k', 'top_p']:
        raise HTTPException(status_code=400, detail="sampling_method 必须是 'beam', 'top_k', 'top_p'")
    if not (1 <= top_k <= 100):
        raise HTTPException(status_code=400, detail="top_k 应在 1~100 之间")
    if not (0.0 < top_p <= 1.0):
        raise HTTPException(status_code=400, detail="top_p 应在 (0, 1] 之间")
    if not (1 <= num_samples <= 100):
        raise HTTPException(status_code=400, detail="num_samples 应在 1~100 之间")
    if not (0.1 <= temperature <= 5.0):
        raise HTTPException(status_code=400, detail="temperature 应在 0.1~5.0 之间")
    '''
    user_id = request.cookies.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="无用户标识")

    if not check_rate_limit(user_id):
        raise HTTPException(status_code=429, detail=f"请求过于频繁，每天最多 {REQUESTS_PER_DAY} 次。")

    if not q or len(q.strip()) == 0:
        raise HTTPException(status_code=400, detail="请输入有效的 SMILES 字符串")

    async def generate_response() -> AsyncGenerator[str, None]:
        try:
            yield "🔍 正在预测逆合成路径...\n"
            mol = Chem.MolFromSmiles(q.strip())
            input_product = Chem.MolToSmiles(mol,canonical=True, kekuleSmiles=False)
            await asyncio.sleep(0.5)

            # 调用化学模型预测前体
            precursors = chem_model.precursor(
                product=input_product,
                top_k=top_k,
                top_p=top_p,
                sampling_method=sampling_method,
                num_samples=num_samples,
                temperature=temperature
            )

            if not precursors or len(precursors) == 0:
                yield "⚠️ 未找到可行的逆合成路径。\n"
                return

            method_name = {
                'beam': 'Beam Search',
                'topk': 'Top-k 采样',
                'nucleus': 'Top-p (Nucleus) 采样'
            }.get(sampling_method, sampling_method)

            yield f"✅ 找到 {len(precursors)} 条可能的合成路径（使用 {method_name}）：\n\n"
            await asyncio.sleep(0.3)

            for i, (precursor_smiles, score) in enumerate(precursors):
                img_tag = ""
                img_url = reaction_smiles_to_image_base64(
                        reactants=[precursor_smiles],
                        reagents=[],
                        products=[q.strip()],
                        size=(800, 400)
                    )
                if img_url:
                    img_tag = f'<br><img src="{img_url}" alt="分子结构" style="max-width: 200px; margin-top: 8px;" />'

                line = f"🧪 方案 {i+1}:\n   前体: {precursor_smiles}\n   置信度: {score:.4f}{img_tag}\n\n"
                yield line
                await asyncio.sleep(0.2)

        except Exception as e:
            yield f"[❌ 预测错误] {str(e)}\n，请输入正确的SMILES表达式"

    return StreamingResponse(generate_response(), media_type="text/plain")


# ========== 5.1 试剂预测接口 ==========
@app.get("/predict/reagent")
async def predict_reagent_endpoint(
    request: Request,
    reactant: str,
    product: str,
    num_samples: int = 3
):
    '''
    if not (1 <= num_samples <= 10):
        raise HTTPException(status_code=400, detail="num_samples 应在 1~10 之间")

    user_id = request.cookies.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="无用户标识")

    if not check_rate_limit(user_id):
        raise HTTPException(status_code=429, detail=f"请求过于频繁，每天最多 {REQUESTS_PER_DAY} 次。")
    '''

    if not reactant or not product:
        raise HTTPException(status_code=400, detail="❌ 请同时输入反应物和产物的 SMILES 表达式")

    async def generate_response() -> AsyncGenerator[str, None]:
        try:
            yield "🧪 正在预测所需试剂...\n"
            mol = Chem.MolFromSmiles(reactant.strip())
            input_reactant = Chem.MolToSmiles(mol,canonical=True, kekuleSmiles=False)
            mol = Chem.MolFromSmiles(product.strip())
            input_product = Chem.MolToSmiles(mol,canonical=True, kekuleSmiles=False)
            await asyncio.sleep(0.5)

            reagents = chem_model.reagent(
                reactant=input_reactant,
                product=input_product,
                n=num_samples
            )

            if not reagents or len(reagents) == 0:
                yield "⚠️ 未找到合适的试剂。\n"
                return

            yield f"✅ 找到 {len(reagents)} 种可能的试剂：\n\n"
            await asyncio.sleep(0.3)

            for i, (reagent_smiles, score) in enumerate(reagents):
                img_tag = ""
                img_url = reaction_smiles_to_image_base64(
                    reactants=[reactant.strip()],
                    reagents=[reagent_smiles],
                    products=[product.strip()]
                )
                if img_url:
                    img_tag = f'<br><img src="{img_url}" alt="分子结构" style="max-width: 200px; margin-top: 8px;" />'

                line = f"🥼 试剂 {i+1}:\n   SMILES: {reagent_smiles}\n   置信度: {score:.4f}{img_tag}\n\n"
                yield line
                await asyncio.sleep(0.2)

        except Exception as e:
            yield f"[❌ 预测错误] {str(e)}\n请检查输入的 SMILES 是否正确"

    return StreamingResponse(generate_response(), media_type="text/plain")


# ========== 5.2 产物预测接口 ==========
@app.get("/predict/product")
async def predict_product_endpoint(
    request: Request,
    reactant: str,
    reagent: str = "",
    num_samples: int = 3
):
    '''
    if not (1 <= num_samples <= 10):
        raise HTTPException(status_code=400, detail="num_samples 应在 1~10 之间")

    user_id = request.cookies.get("user_id")
    if not user_id:
        raise HTTPException(status_code=400, detail="无用户标识")

    if not check_rate_limit(user_id):
        raise HTTPException(status_code=429, detail=f"请求过于频繁，每天最多 {REQUESTS_PER_DAY} 次。")
    '''

    if not reactant:
        raise HTTPException(status_code=400, detail="请输入有效的反应物 SMILES")

    async def generate_response() -> AsyncGenerator[str, None]:
        try:
            yield "🧪 正在预测反应产物...\n"
            mol = Chem.MolFromSmiles(reactant.strip())
            input_reactant = Chem.MolToSmiles(mol,canonical=True, kekuleSmiles=False)
            mol = Chem.MolFromSmiles(reagent.strip())
            input_reagent = Chem.MolToSmiles(mol,canonical=True, kekuleSmiles=False)
            await asyncio.sleep(0.5)

            products = chem_model.product(
                reactant=input_reactant,
                reagent=input_reagent if reagent else None,
                n=num_samples
            )

            if not products or len(products) == 0:
                yield "⚠️ 未找到可能的产物。\n"
                return

            yield f"✅ 找到 {len(products)} 种可能的产物：\n\n"
            await asyncio.sleep(0.3)

            for i, (product_smiles, score) in enumerate(products):
                img_tag = ""
                reagents = [reagent.strip()] if reagent.strip() else []
                img_url = reaction_smiles_to_image_base64(
                    reactants=[reactant.strip()],
                    reagents=reagents,
                    products=[product_smiles]
                )
                if img_url:
                    img_tag = f'<br><img src="{img_url}" alt="分子结构" style="max-width: 200px; margin-top: 8px;" />'

                line = f"🧪 产物 {i+1}:\n   SMILES: {product_smiles}\n   置信度: {score:.4f}{img_tag}\n\n"
                yield line
                await asyncio.sleep(0.2)

        except Exception as e:
            yield f"[❌ 预测错误] {str(e)}\n请检查输入的 SMILES 是否正确"

    return StreamingResponse(generate_response(), media_type="text/plain")



# ========== 启动服务 ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)