"""
最小 MCP Server。

目前僅建立 MCP 連線，作為 Host 端保留的通訊入口。
後續若加入路由層、工具呼叫或外部資源，可在此擴充。
"""

from fastmcp import FastMCP

# 建立 MCP Server。
mcp = FastMCP("Campus Agent Entrypoint")


if __name__ == "__main__":
    # 使用 stdio 傳輸啟動，供 Streamlit 端以 Client 連線。
    mcp.run()
