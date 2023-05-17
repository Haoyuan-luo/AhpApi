from django.shortcuts import render

# Create your views here.
from django.shortcuts import render

from django.http.response import JsonResponse
from rest_framework.parsers import JSONParser
from rest_framework import status
import pandas as pd
import numpy as np
import os
from AnalyticHierarchyProcess.ahp_model import AHP as CalcModel
from rest_framework.decorators import api_view

current_path = os.path.abspath(__file__)
parent_path = os.path.dirname(os.path.dirname(current_path)) + "/AnalyticHierarchyProcess"


@api_view(['GET', 'POST', 'DELETE'])
def CalculateAhp(request):
    if request.method == 'GET':
        version = request.GET.get('version')

        targetFileName = parent_path + "/" + 'calc_matrix_' + version + '.xlsx'
        guildFileName = parent_path + "/" + "guild.xlsx"

        if os.path.isfile(targetFileName) and os.path.isfile(guildFileName):
            # 读取准则矩阵
            guildSheet = pd.read_excel(guildFileName)
            gCriteria = np.array([row for _, row in guildSheet.iterrows()])
            # 读取方案矩阵
            writer = pd.ExcelFile(targetFileName)
            sheet_len = len(writer.sheet_names)
            # 指定下标读取
            tCriteria = []
            for i in range(0, sheet_len):
                targetSheet = pd.read_excel(writer, sheet_name=i)
                tCriteria.append(np.array([row for _, row in targetSheet.iterrows()]))

            # 对方案矩阵进行计算
            ret = CalcModel(gCriteria, tCriteria).run()

            if not isinstance(ret, np.ndarray):
                response = JsonResponse({'business message': "internal business error"},
                                        json_dumps_params={'ensure_ascii': False})
                response['Content-Type'] = 'application/json;charset=utf-8'
                return response
            else:
                print(f"calculate version {version}")
                response = JsonResponse({'business message': "calculate matrix done", 'result': ret.max()},
                                        json_dumps_params={'ensure_ascii': False})
                response['Content-Type'] = 'application/json;charset=utf-8'

            return response

        else:
            print("no exist target matrix to calculate")
            response = JsonResponse({'result': "no exist target matrix to calculate"},
                                    json_dumps_params={'ensure_ascii': False})
            response['Content-Type'] = 'application/json;charset=utf-8'
            return response
