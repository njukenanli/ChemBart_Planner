from rdkit import Chem

class utils():
    @staticmethod
    def canonize(smi, allow_error = False):
        smilist = smi.split(".")
        newlist = []
        for i in smilist:
            if i == "":
                continue
            mol = Chem.MolFromSmiles(i)
            if mol is None:
                if allow_error:
                    continue
                else:
                    return None
            else:
                part = Chem.MolToSmiles(mol,canonical=True, kekuleSmiles=False)
                newlist.append(part)
        if not newlist:
            return None
        return ".".join(newlist)

    @staticmethod
    def weak_compare(str1, str2):
        l1 = str1.split(".")
        l2 = str2.split(".")
        for i in l1:
            for j in l2:
                if i == j:
                    return True
        return False

    @staticmethod
    def general_basic_mol(canonized_smi):
        mol = Chem.MolFromSmiles(canonized_smi)
        if mol is None:
            return 0
        c = 0
        for i in mol.GetAtoms():
            if i.GetAtomicNum() == 6:
                c += 1
        return (c<=3)


# retro_analysis.py
import os
import re
import json
import argparse
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import logging

import pandas as pd
import matplotlib.pyplot as plt
from rdkit import Chem
from rdkit.Chem import Draw
from io import BytesIO
from PIL import Image
import numpy as np
import cv2

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class AnalysisResult:
    """数据类存储分析结果"""
    mol_id: str
    steps: int
    success: int
    route_number: int
    reaction_data: Dict
    route_path: List[str] = None
    

class RetroAnalyzer:
    """逆合成分析工具类"""
    def __init__(self, target_dir: str, output_csv: str):
        self.target_dir = Path(target_dir)
        self.output_csv = Path(output_csv)
        self.results = []

    def analyze_directory(self) -> pd.DataFrame:
        """分析目录下的所有文件"""
        logger.info(f"开始分析目录: {self.target_dir}")
        
        # 修改为匹配所有以数字结尾的文件
        files = []
        for f in self.target_dir.glob('*_*_route_*'):
            # 验证文件名格式是否符合预期
            if re.match(r'.+_route_\d+$', f.stem):
                files.append(f)
        
        # 按路线号排序
        files.sort(key=lambda x: int(re.search(r'_(\d+)$', x.stem).group(1)))
        
        # 并行分析文件（可根据需要启用多进程）
        for file_path in files:
            logger.debug(f"分析文件: {file_path}")
            result = self.analyze_file(file_path)
            if result:
                self.results.append(result)
        
        # 生成CSV报告
        df = self.generate_report()
        df.to_csv(self.output_csv, index=False)
        logger.info(f"分析完成，结果已保存至: {self.output_csv}")
        return df

    def analyze_file(self, file_path: Path) -> Optional[AnalysisResult]:
        """分析单个文件"""
        try:
            # 提取分子ID
            mol_id = self._extract_molecule_id(file_path)

            # 提取路线号
            route_number = self._extract_route_number(file_path)
            
            # 解析反应信息
            reaction_data, success = self._parse_reaction_data(file_path)
            steps = len(reaction_data)
            if 'basic mol' in reaction_data.values():
                steps = steps -1 

            # 获取路径信息
            if success == 1:
                route_path = self._get_route_path(reaction_data)
            else:
                route_path = []
            
            return AnalysisResult(
                mol_id=mol_id,
                steps=steps,
                success=success,
                reaction_data=reaction_data,
                route_path=route_path,
                route_number=route_number
            )
        except Exception as e:
            logger.error(f"分析文件失败 {file_path}: {str(e)}")
            return None

    def _extract_molecule_id(self, file_path: Path) -> str:
        """提取分子ID"""
        filename = file_path.stem
        #print(filename)
        match = re.search(r'_([^_]+)_route_', filename)
        if not match:
            raise ValueError(f"无法从文件名提取分子ID: {filename}")
        return match.group(1)

    def _extract_route_number(self, file_path: Path) -> int:
        """新增：提取路线号"""
        filename = file_path.stem
        match = re.search(r'_route_(\d+)$', filename)
        if not match:
            raise ValueError(f"无法从文件名提取路线号: {filename}")
        return int(match.group(1))

    def _parse_reaction_data(self, file_path: Path) -> Tuple[Dict, int]:
        """提取单步信息，解析反应是否成功"""
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 提取成功状态
        success = 1 if '"success": 1' in content else 0
        
        # 分割部分
        sections = re.split(r'(?:synthesis route:|reaction information per step:)', content)
        
        return json.loads(sections[-1].strip()), success

    def _get_route_path(self, reaction_data: Dict) -> List[str]:
        # 读取反应信息
        path = []

        if not reaction_data:
            return path
        # 获取起始分子（即目标分子）
        current_mol = next(iter(reaction_data))
        path.append(current_mol)

        # 依次向下查找 precursors，构建路径
        try:
            for current_mol in reaction_data:
                precursors = reaction_data[current_mol].get('precursors', '')
                path.append(precursors)
        
        except:
            return path
            #print()
       
        return path
    
    def _get_reagent_path(self, reaction_data: Dict) -> List[str]:
        # 读取反应信息
        reagent_list = []
        temp_list = []
        yield_list = []

        # 依次向下查找 precursors，构建路径
        try:
            for current_mol in reaction_data:
                reagent = reaction_data[current_mol].get('reagents', '')
                reagent_list.append(reagent)
                temp = reaction_data[current_mol].get('temperature', '')
                temp_list.append(temp)
                yield_0 = reaction_data[current_mol].get('yield', '')
                yield_list.append(yield_0)

        except:
            return None
            #print()
       
        return reagent_list,temp_list,yield_list
    
    
    def generate_report(self) -> pd.DataFrame:
        """生成分析报告"""
        data = [{
            'mol_pro': r.mol_id,
            'route_number': r.route_number,
            'steps': r.steps,
            'success': r.success,
            'route': json.dumps(r.route_path)
        } for r in self.results if r]
        
        df = pd.DataFrame(data)
        return df.sort_values(['mol_pro', 'route_number'])

class Visualizer:
    """可视化工具类"""

    @staticmethod
    def visualize_route_only_mols(path: List[str], output_path: Optional[Path] = None) -> None:
        """
        可视化分子路径（仅分子结构）
        :param path: 分子 SMILES 路径（从目标分子到起始分子）
        :param output_path: 输出图像路径（可选）
        """
        try:
            mols = [Chem.MolFromSmiles(smi) for smi in reversed(path)]

            legends = [f"Step {i}" for i, smi in enumerate(path)]

            img = Draw.MolsToGridImage(
                mols,
                molsPerRow=4,
                subImgSize=(400, 400),
                highlightAtomLists=None,
                legends=legends
            )

            if output_path:
                img.save(output_path)
                logger.info(f"分子路径图已保存至: {output_path}")
            else:
                plt.figure(figsize=(12, 8))
                plt.imshow(img)
                plt.axis('off')
                plt.show()

        except Exception as e:
            logger.error(f"分子路径可视化失败: {str(e)}")
    


    @staticmethod
    def visualize_reaction_conditions(
        reagent_list: List[List[str]],
        temp_list: List[str],
        yield_list: List[str],
        output_path: Optional[Path] = None
    ) -> None:
        """
        可视化每一步反应中的试剂（用分子结构展示），并在下方显示温度和产率。
        :param reagent_list: 每步反应的试剂列表（每个元素是该步所有试剂的列表）
        :param temp_list: 每步反应的温度（字符串列表）
        :param yield_list: 每步反应的产率（字符串列表）
        :param output_path: 输出图像路径（可选）
        """
        reagent_list = reagent_list[::-1]
        temp_list = temp_list[::-1]
        yield_list = yield_list[::-1]
        mols = [Chem.MolFromSmiles(smi) for smi in reagent_list]

        legends = []

        def format_value(val) -> str:

            """将数值格式化为最多两位有效数字，非数字返回原值"""
            try:
                # 尝试转换为浮点数
                val_float = float(val)
                return "{:.2g}".format(val_float)
            except (ValueError, TypeError):
                return str(val)
            
        for step_idx, (temp, yield_) in enumerate(zip(temp_list, yield_list)):
            formatted_temp = format_value(temp)
            formatted_yield = format_value(yield_)
            
            legend_text = f"Step {step_idx+1} Temp: {formatted_temp} Yield: {formatted_yield}"
            legends.append(legend_text)


        # 绘制图像网格
        img = Draw.MolsToGridImage(
            mols=mols,
            molsPerRow=4,
            legends=legends,
            subImgSize=(300, 300)
        )

        if output_path:
            img.save(output_path)
            logger.info(f"反应条件图已保存至: {output_path}")
        else:
            plt.figure(figsize=(12, 8))
            plt.imshow(img)
            plt.axis('off')
            plt.tight_layout()
            plt.show()

