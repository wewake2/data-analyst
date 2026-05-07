"""
Query Price Calculator

Estimate the cost of LLM queries based on token usage and
AWS Bedrock pricing.
"""
import streamlit as st

st.set_page_config(page_title="Query Price Calculator", layout="centered")

st.title("Query Price Calculator")
st.caption("Estimate the cost of your queries based on token usage and model pricing.")

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    input_tokens = st.number_input(
        "Input tokens",
        min_value=0, value=0, step=100,
        help="Total input tokens consumed by the query",
    )
with col2:
    output_tokens = st.number_input(
        "Output tokens",
        min_value=0, value=0, step=100,
        help="Total output tokens consumed by the query",
    )

st.markdown("### Pricing (per 1M tokens)")
price_col1, price_col2 = st.columns(2)
with price_col1:
    input_price = st.number_input(
        "Input token price (USD / 1M tokens)",
        min_value=0.0, value=3.0, step=0.01, format="%.4f",
    )
with price_col2:
    output_price = st.number_input(
        "Output token price (USD / 1M tokens)",
        min_value=0.0, value=15.0, step=0.01, format="%.4f",
    )

st.markdown("### Currency conversion")
usd_to_inr = st.number_input(
    "USD to INR rate",
    min_value=0.0, value=84.91, step=0.01, format="%.2f",
)

st.markdown("---")

input_cost_usd = (input_tokens / 1_000_000) * input_price
output_cost_usd = (output_tokens / 1_000_000) * output_price
total_cost_usd = input_cost_usd + output_cost_usd
total_cost_inr = total_cost_usd * usd_to_inr

st.markdown("### Cost breakdown")

breakdown_col1, breakdown_col2, breakdown_col3 = st.columns(3)
with breakdown_col1:
    st.metric("Input cost", f"${input_cost_usd:.6f}")
with breakdown_col2:
    st.metric("Output cost", f"${output_cost_usd:.6f}")
with breakdown_col3:
    st.metric("Total (USD)", f"${total_cost_usd:.6f}")

st.markdown("---")

result_col1, result_col2 = st.columns(2)
with result_col1:
    st.metric("Total cost (USD)", f"${total_cost_usd:.6f}")
with result_col2:
    st.metric("Total cost (INR)", f"Rs. {total_cost_inr:.4f}")
