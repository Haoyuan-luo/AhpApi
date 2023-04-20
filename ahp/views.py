from django.shortcuts import render

# Create your views here.
from django.http import JsonResponse
import numpy as np


def ahp_view(request):
    A = request.GET.getlist('A')
    A = [float(x) for x in A]
    A = [A[i:i + 3] for i in range(0, len(A), 3)]
    A = np.array(A)
    N = len(A)

    # 判断矩阵检验
    m, n = np.linalg.eig(A)
    lamdamax = max(np.diag(n))
    CI = (lamdamax - N) / (N - 1)
    RILIST = [0, 0, 0.58, 0.9, 1.12, 1.24, 1.32, 1.41, 1.46]
    RI = RILIST[N]
    CR = CI / RI
    if CR < 0.1:
        result = "判断矩阵合理"
    else:
        result = "判断矩阵不合理"

    # 权重求解
    a = np.ones((N, 1))
    for i in range(N):
        for j in range(N):
            a[i, 0] *= A[i, j]
    W = np.power(a, 1 / N)
    w = W / np.sum(W)

    response = JsonResponse({'result': result, 'w': w.tolist()}, json_dumps_params={'ensure_ascii': False})
    response['Content-Type'] = 'application/json;charset=utf-8'
    return response


# 更多的可用来预测的判别模型
def prediction_model():
    pass
