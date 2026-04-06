# main.py - Chemical Retrosynthesis Prediction Service (Based on CBRetro)
import uuid
import time
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
from utils.show_smiles import reaction_smiles_to_image_base64
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)
from ChemBart4web import CBRetro



# ========== 1. Initialize Chemical Model ==========
print("Loading chemical retrosynthesis model...")
MODEL_PATH = "../retrain_pv_temp/model/ChemBart_Full4.pth"  # ⚠️ Please replace with your actual model path
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

try:
    chem_model = CBRetro(path=MODEL_PATH, dev=DEVICE)
    print("Chemical model loaded successfully!")
except Exception as e:
    print(f"Model loading failed: {e}")
    raise

# ========== 2. Create FastAPI Application ==========
app = FastAPI(title="ChemBart Retrosynthesis Assistant")
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="static")


# ========== 3. Homepage Route ==========
@app.get("/", response_class=HTMLResponse)
async def get_chat_page(request: Request):
    # Set user_id cookie for session continuity (optional, not used for quota)
    user_id = request.cookies.get("user_id")
    if not user_id:
        user_id = str(uuid.uuid4())
    response = templates.TemplateResponse("single_step_en.html", {"request": request})
    response.set_cookie(key="user_id", value=user_id, httponly=True, max_age=86400*30)
    return response


# ========== 4. Core Retrosynthesis Prediction Endpoint (Streaming) ==========
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
    # Input validation
    if not q or len(q.strip()) == 0:
        raise HTTPException(status_code=400, detail="Please enter a valid SMILES string")

    async def generate_response() -> AsyncGenerator[str, None]:
        try:
            yield "🔍 Predicting retrosynthetic pathways...\n"
            mol = Chem.MolFromSmiles(q.strip())
            if mol is None:
                yield "[❌ Error] Invalid SMILES expression. Please check your input.\n"
                return
            input_product = Chem.MolToSmiles(mol, canonical=True, kekuleSmiles=False)
            await asyncio.sleep(0.5)

            # Call chemical model to predict precursors
            precursors = chem_model.precursor(
                product=input_product,
                top_k=top_k,
                top_p=top_p,
                sampling_method=sampling_method,
                num_samples=num_samples,
                temperature=temperature
            )

            if not precursors or len(precursors) == 0:
                yield "⚠️ No feasible retrosynthetic pathways found.\n"
                return

            method_name = {
                'beam': 'Beam Search',
                'top_k': 'Top-k Sampling',
                'top_p': 'Top-p (Nucleus) Sampling'
            }.get(sampling_method, sampling_method)

            yield f"✅ Found {len(precursors)} possible synthetic routes (using {method_name}):\n\n"
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
                    img_tag = f'<br><img src="{img_url}" alt="Molecular structure" style="max-width: 200px; margin-top: 8px;" />'

                line = f"🧪 Route {i+1}:\n   Precursor: {precursor_smiles}\n   Confidence: {score:.4f}{img_tag}\n\n"
                yield line
                await asyncio.sleep(0.2)

        except Exception as e:
            yield f"[❌ Prediction error] {str(e)}\nPlease verify your SMILES expression is valid."

    return StreamingResponse(generate_response(), media_type="text/plain")


# ========== 4.1 Reagent Prediction Endpoint ==========
@app.get("/predict/reagent")
async def predict_reagent_endpoint(
    request: Request,
    reactant: str,
    product: str,
    num_samples: int = 3
):
    if not reactant or not product:
        raise HTTPException(status_code=400, detail="❌ Please provide both reactant and product SMILES expressions")

    async def generate_response() -> AsyncGenerator[str, None]:
        try:
            yield "🧪 Predicting required reagents...\n"
            
            mol_r = Chem.MolFromSmiles(reactant.strip())
            mol_p = Chem.MolFromSmiles(product.strip())
            if mol_r is None or mol_p is None:
                yield "[❌ Error] Invalid SMILES expression(s). Please check your inputs.\n"
                return
                
            input_reactant = Chem.MolToSmiles(mol_r, canonical=True, kekuleSmiles=False)
            input_product = Chem.MolToSmiles(mol_p, canonical=True, kekuleSmiles=False)
            await asyncio.sleep(0.5)

            reagents = chem_model.reagent(
                reactant=input_reactant,
                product=input_product,
                n=num_samples
            )

            if not reagents or len(reagents) == 0:
                yield "⚠️ No suitable reagents found.\n"
                return

            yield f"✅ Found {len(reagents)} possible reagents:\n\n"
            await asyncio.sleep(0.3)

            for i, (reagent_smiles, score) in enumerate(reagents):
                img_tag = ""
                img_url = reaction_smiles_to_image_base64(
                    reactants=[reactant.strip()],
                    reagents=[reagent_smiles],
                    products=[product.strip()]
                )
                if img_url:
                    img_tag = f'<br><img src="{img_url}" alt="Molecular structure" style="max-width: 200px; margin-top: 8px;" />'

                line = f"🥼 Reagent {i+1}:\n   SMILES: {reagent_smiles}\n   Confidence: {score:.4f}{img_tag}\n\n"
                yield line
                await asyncio.sleep(0.2)

        except Exception as e:
            yield f"[❌ Prediction error] {str(e)}\nPlease verify your SMILES expressions are valid."

    return StreamingResponse(generate_response(), media_type="text/plain")


# ========== 4.2 Product Prediction Endpoint ==========
@app.get("/predict/product")
async def predict_product_endpoint(
    request: Request,
    reactant: str,
    reagent: str = "",
    num_samples: int = 3
):
    if not reactant:
        raise HTTPException(status_code=400, detail="Please enter a valid reactant SMILES")

    async def generate_response() -> AsyncGenerator[str, None]:
        try:
            yield "🧪 Predicting reaction products...\n"
            
            mol_r = Chem.MolFromSmiles(reactant.strip())
            if mol_r is None:
                yield "[❌ Error] Invalid reactant SMILES. Please check your input.\n"
                return
            input_reactant = Chem.MolToSmiles(mol_r, canonical=True, kekuleSmiles=False)
            
            input_reagent = None
            if reagent.strip():
                mol_rg = Chem.MolFromSmiles(reagent.strip())
                if mol_rg is None:
                    yield "[❌ Error] Invalid reagent SMILES. Please check your input.\n"
                    return
                input_reagent = Chem.MolToSmiles(mol_rg, canonical=True, kekuleSmiles=False)
            
            await asyncio.sleep(0.5)

            products = chem_model.product(
                reactant=input_reactant,
                reagent=input_reagent if reagent.strip() else None,
                n=num_samples
            )

            if not products or len(products) == 0:
                yield "⚠️ No possible products found.\n"
                return

            yield f"✅ Found {len(products)} possible products:\n\n"
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
                    img_tag = f'<br><img src="{img_url}" alt="Molecular structure" style="max-width: 200px; margin-top: 8px;" />'

                line = f"🧪 Product {i+1}:\n   SMILES: {product_smiles}\n   Confidence: {score:.4f}{img_tag}\n\n"
                yield line
                await asyncio.sleep(0.2)

        except Exception as e:
            yield f"[❌ Prediction error] {str(e)}\nPlease verify your SMILES expressions are valid."

    return StreamingResponse(generate_response(), media_type="text/plain")


# ========== 5. Start Server ==========
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)