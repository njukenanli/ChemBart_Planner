from rdkit import Chem
from rdkit.Chem import Draw
import base64
import io

def reaction_smiles_to_image_base64(reactants: list, reagents: list, products: list, size=(400, 200)) -> str:
    """
    将反应物、试剂、产物 SMILES 列表转换为反应式图片（Base64）
    修复 RDKit DrawReaction 参数兼容性问题
    """
    try:
        from rdkit.Chem import AllChem
        from rdkit.Chem.Draw import MolDraw2DCairo, DrawingOptions
        from rdkit.Chem.Draw.rdMolDraw2D import MolDraw2D
        from rdkit.Chem import rdChemReactions

        # 创建空反应
        rxn = AllChem.ChemicalReaction()

        # 添加反应物
        for smi in reactants:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                rxn.AddReactantTemplate(mol)

        # 添加试剂（RDKit 中试剂也作为反应物添加）
        for smi in reagents:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                rxn.AddReactantTemplate(mol)

        # 添加产物
        for smi in products:
            mol = Chem.MolFromSmiles(smi)
            if mol:
                rxn.AddProductTemplate(mol)

        if rxn.GetNumReactantTemplates() == 0 and rxn.GetNumProductTemplates() == 0:
            return ""

        # 手动创建绘图对象
        drawer = MolDraw2DCairo(size[0], size[1])
        opts = drawer.drawOptions()
        opts.padding = 0.1  # 适当留白

        # ✅ 关键修复：显式调用 DrawReaction，避免默认参数冲突
        drawer.DrawReaction(rxn, highlightByReactant=False)

        drawer.FinishDrawing()

        # 获取 PNG 数据
        png_data = drawer.GetDrawingText()
        img_str = base64.b64encode(png_data).decode("utf-8")
        return f"data:image/png;base64,{img_str}"

    except Exception as e:
        print(f"生成反应式图片失败: {e}")
        return ""

def smiles_to_image_base64(smiles: str, size=(200, 150)) -> str:
    """
    将 SMILES 转换为 Base64 编码的 PNG 图片
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""  # 无效 SMILES

        img = Draw.MolToImage(mol, size=size)
        buffer = io.BytesIO()
        img.save(buffer, format="PNG")
        img_str = base64.b64encode(buffer.getvalue()).decode("utf-8")
        return f"data:image/png;base64,{img_str}"
    except Exception as e:
        print(f"生成分子图片失败: {e}")
        return ""