# ethapp/views.py
from django.shortcuts import render
from django.http import JsonResponse
from .ml import get_eth_advice
from .smart_contract import user_has_paid
import random

def home(request):
    return render(request, "index.html")

def get_advice(request):
    try:
        eth_address = request.GET.get('address')
        if not eth_address:
            return JsonResponse({"error": "Missing Ethereum address"}, status=400)

        result = get_eth_advice()  # call your PyTorch model
        return JsonResponse(result)  # send all info to frontend as JSON

    except Exception as e:
        print(" Django Error:", e)
        return JsonResponse({"error": str(e)}, status=500)
