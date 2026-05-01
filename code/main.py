
import pandas as pd

from utils import combine_text
from classifier import classify_request, classify_product_area, assess_risk
from retriever import Retriever
from decision import decide_action
from response_generator import generate_response, generate_justification
from logger import log


def main():
    input_path = "../support_issues/support_issues.csv"
    output_path = "../support_issues/output.csv"

    df = pd.read_csv(input_path)

    retriever = Retriever()

    results = []

    for i, row in df.iterrows():
        log(f"[INFO] Processing row {i}")

        text = combine_text(row.get("subject", ""), row.get("issue", ""))
        company = str(row.get("company", ""))

        request_type = classify_request(text)
        product_area = classify_product_area(text, company)
        risk = assess_risk(text)

        doc, confidence = retriever.search(text)

        status = decide_action(risk, confidence)

        response = generate_response(status, doc)
        justification = generate_justification(
            request_type, risk, status, confidence
        )

        log(f"[TYPE] {request_type}")
        log(f"[AREA] {product_area}")
        log(f"[RISK] {risk}")
        log(f"[CONF] {confidence}")
        log(f"[STATUS] {status}")

        results.append({
            "status": status,
            "product_area": product_area,
            "response": response,
            "justification": justification,
            "request_type": request_type
        })

    output_df = pd.DataFrame(results)
    output_df.to_csv(output_path, index=False)

    print("✅ Output generated at support_issues/output.csv")


if __name__ == "__main__":
    main()
