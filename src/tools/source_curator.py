from typing import List, Dict
import urllib.parse


class SourceCurator:
    def __init__(self):
        self.max_candidates = 6

    def _extract_tokens(self, query: str) -> List[str]:
        q = (query or "").strip()
        tokens = [t for t in q.replace("/", " ").replace("-", " ").split() if t]
        # 仅保留 ASCII 词元，避免中文被误用为域名
        import re
        tokens = [t for t in tokens if re.fullmatch(r"[A-Za-z0-9_\-]{2,20}", t) is not None]
        return list(dict.fromkeys(tokens))

    def curate(self, query: str) -> List[Dict[str, str]]:
        tokens = self._extract_tokens(query)
        candidates: List[Dict[str, str]] = []
        ql = (query or "").lower()
        # 公司/品牌定制源（提高有效内容比例）
        if any(k in ql for k in ["字节跳动", "bytedance"]):
            candidates.extend([
                {"name": "bytedance_cn", "url": "https://www.bytedance.com/zh", "description": "字节跳动官网(中文)"},
                {"name": "bytedance_en", "url": "https://www.bytedance.com/en", "description": "ByteDance Official"},
                {"name": "tiktok_newsroom", "url": "https://newsroom.tiktok.com/", "description": "TikTok 新闻室"},
                {"name": "lark_blog", "url": "https://www.larksuite.com/blog", "description": "飞书(Lark) 博客"},
            ])
        # 取消通用域名猜测，避免无效URL
        # 通用产品与生态页
        candidates.append({"name": "product_wikipedia", "url": f"https://zh.wikipedia.org/wiki/Special:Search?search={urllib.parse.quote(query)}", "description": "百科概览"})
        # 去重并截断
        seen = set()
        uniq: List[Dict[str, str]] = []
        for c in candidates:
            u = c["url"]
            if u in seen:
                continue
            seen.add(u)
            uniq.append(c)
            if len(uniq) >= self.max_candidates:
                break
        return uniq
