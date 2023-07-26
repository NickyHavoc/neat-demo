import requests

from typing import List

from ..tool import Tool, ToolParam, ToolResult


TOOL_PARAM_STOCK_TRADING_SYMBOL = ToolParam(
    name="stock_trading_symbol",
    type="string",
    description="The stock trading symbol of a company (e.g. AAPL for Apple).",
    required=True
)


class FinancialRetrievalTool(Tool):
    def __init__(
        self,
        alpha_vantage_api_key: str,
        name: str = "Stock Trading API",
        description: str = "Retrieve the current weather for a location.",
        params: List[ToolParam] = [
            TOOL_PARAM_STOCK_TRADING_SYMBOL
        ]
    ):
        super().__init__(name, description, params)
        self.api_key = alpha_vantage_api_key
        self.url = "https://www.alphavantage.co/query?function=TIME_SERIES_INTRADAY&symbol={symbol}&interval=5min&apikey={api_key}"

    def run(self, json_query: dict) -> ToolResult:
        self.legal_params(json_query)
        symbol = json_query["stock_trading_symbol"]
        url = self.url.format(
            symbol=symbol,
            api_key=self.api_key
        )
        response = requests.get(url)
        response_json = response.json()
        open_value = response_json["Time Series (5min)"][list(
            response_json["Time Series (5min)"].keys())[-1]]["1. open"]
        close_value = response_json["Time Series (5min)"][list(
            response_json["Time Series (5min)"].keys())[0]]["4. close"]
        return self._build_tool_result(
            [f"open_value: {open_value}", f"close_value: {close_value}"])
