# services/recommendation.py
import requests

def get_deepseek_recommendations(bill_data):
    """
    Hypothetical function that queries a 'deepseek R1' backend for custom tips.
    We'll just mock a response based on usage or cost from the PDF parse.
    """
    if not bill_data:
        return []
    usage = bill_data.get("usage_kWh", 0)
    cost  = bill_data.get("bill_cost", 0)
    recs= []
    if usage>800:
        recs.append("Shift heavy loads to off-peak to lower usage during peak hours.")
    if cost>150:
        recs.append("Investigate solar or battery to offset high monthly bills.")
    if usage>1200 and cost>200:
        recs.append("Large usage suggests checking if a demand-rate plan might be cheaper with battery backup.")
    return recs
