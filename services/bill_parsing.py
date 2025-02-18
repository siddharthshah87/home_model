# services/bill_parsing.py
import PyPDF2
import re

def parse_utility_bill_pdf(uploaded_file):
    """
    A naive PDF parsing approach. 
    We'll attempt to read text, find lines like 'Total Usage: 1234' or 'Total Cost: $123.45'.
    """
    try:
        reader = PyPDF2.PdfReader(uploaded_file)
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text() or ""

        usage_match = re.search(r"Total Usage:\s+(\d+)", full_text)
        cost_match = re.search(r"Total Cost:\s+\$?(\d+\.\d+)", full_text)
        result={}
        if usage_match:
            result["usage_kWh"]= int(usage_match.group(1))
        if cost_match:
            result["bill_cost"]= float(cost_match.group(1))
        return result if result else None
    except Exception as e:
        print(f"Error parsing PDF: {e}")
        return None
