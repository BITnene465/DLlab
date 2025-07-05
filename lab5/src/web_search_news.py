# TODO： 知乎、今日头条的文章抓不到

import requests
import re
import os
import time
import random
import logging
import json
from datetime import datetime
from typing import List, Dict, Optional, Set

from duckduckgo_search import DDGS
import trafilatura

# 配置日志记录
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('news_spider.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)


class NewsSpider:
    """新闻爬虫类，支持搜索和提取新闻内容"""
    
    def __init__(self, output_dir: str = "news_data", news_id_start: int = 1):
        """初始化新闻爬虫"""
        self.output_dir = output_dir
        self.news_id = news_id_start
        self.crawled_urls: Set[str] = set()
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        # 配置请求会话
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
        
        # 加载已爬取的URL
        self._load_crawled_urls()
    
    def _load_crawled_urls(self) -> None:
        """加载已爬取的URL列表"""
        try:
            if os.path.exists(f"{self.output_dir}/crawled_urls.json"):
                with open(f"{self.output_dir}/crawled_urls.json", 'r', encoding='utf-8') as f:
                    self.crawled_urls = set(json.load(f))
        except Exception as e:
            logging.warning(f"加载已爬取URL列表失败: {e}")
    
    def _save_crawled_urls(self) -> None:
        """保存已爬取的URL列表"""
        try:
            with open(f"{self.output_dir}/crawled_urls.json", 'w', encoding='utf-8') as f:
                json.dump(list(self.crawled_urls), f, ensure_ascii=False, indent=2)
        except Exception as e:
            logging.error(f"保存已爬取URL列表失败: {e}")

    def search_news_links(self, query: str, max_results: int = 20) -> List[str]:
        """使用DuckDuckGo搜索新闻链接"""
        logging.info(f"开始搜索: '{query}'")
        
        try:
            with DDGS(timeout=30) as ddgs:
                results = ddgs.text(query, max_results=max_results)
                if results:
                    links = [r['href'] for r in results if self._is_valid_news_url(r['href'])]
                    logging.info(f"搜索到 {len(links)} 个有效链接")
                    return links
        except Exception as e:
            logging.error(f"搜索失败: {query}, 错误: {e}")
        
        return []

    def _is_valid_news_url(self, url: str) -> bool:
        """检查URL是否为有效的新闻链接"""
        invalid_domains = ['youtube.com', 'twitter.com', 'facebook.com', 'instagram.com']
        invalid_extensions = ['.pdf', '.doc', '.docx', '.xls', '.xlsx']
        
        url_lower = url.lower()
        return not any(domain in url_lower for domain in invalid_domains) and \
               not any(url_lower.endswith(ext) for ext in invalid_extensions)
    
    def extract_news_content(self, url: str) -> Optional[Dict]:
        """提取新闻内容"""
        try:
            logging.info(f"正在提取内容: {url}")
            
            # 使用requests获取页面内容
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            
            # 使用trafilatura提取内容
            json_output_str = trafilatura.extract(
                response.text,
                output_format='json',
                favor_precision=True,
                include_comments=False,
                include_images=False,
                include_tables=True
            )
            
            if not json_output_str:
                logging.warning(f"未能从 {url} 提取到内容")
                return None
            
            metadata = json.loads(json_output_str)
            
            if metadata and metadata.get('text'):
                # 清理和处理文本
                metadata['text'] = self._clean_text(metadata['text'])
                metadata['url'] = url
                metadata['crawl_time'] = datetime.now().isoformat()
                return metadata
            else:
                logging.warning(f"提取的内容为空: {url}")
                return None
                
        except Exception as e:
            logging.error(f"提取内容时发生错误: {url}, 错误: {e}")
            return None
    
    def _clean_text(self, text: str) -> str:
        """清理文本内容"""
        if not text:
            return ""
        
        # 移除多余的空白字符
        text = re.sub(r'\n\s*\n', '\n', text)
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()
        
        return text.strip()
    
    def save_news(self, metadata: Dict, query: str) -> bool:
        """保存新闻为JSON格式"""
        content = metadata.get('text', '')
        if not content or len(content) < 150:
            logging.warning(f"内容过短或为空，跳过保存. URL: {metadata.get('url', 'Unknown')}")
            return False
        
        try:
            # 准备保存的数据
            news_data = {
                "id": self.news_id,
                "title": metadata.get('title', 'N/A'),
                "content": content,
                "author": metadata.get('author', 'N/A'),
                "publish_date": metadata.get('date', 'N/A'),
                "url": metadata.get('url', ''),
                "search_query": query,
                "crawl_time": metadata.get('crawl_time', datetime.now().isoformat()),
                "word_count": len(content),
                "site_name": metadata.get('sitename', 'N/A'),
                "language": metadata.get('language', 'zh')
            }
            
            # 保存为JSON文件
            filename = f"{self.output_dir}/news_{self.news_id:06d}.json"
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(news_data, f, ensure_ascii=False, indent=2)
            
            logging.info(f"保存成功: {filename} (标题: {news_data['title'][:50]}...)")
            self.news_id += 1
            return True
            
        except Exception as e:
            logging.error(f"保存文件失败: {e}")
            return False
    
    def crawl_news(self, search_queries: List[str], max_news_per_query: int = 5, min_total_news: int = 50) -> Dict:
        """主爬虫函数"""
        start_time = datetime.now()
        total_saved = 0
        
        logging.info(f"开始爬取新闻，目标: {min_total_news} 篇")
        
        for query in search_queries:
            if total_saved >= min_total_news:
                logging.info(f"已达到目标新闻数量({min_total_news}篇)，停止爬取")
                break
            
            saved_for_query = 0
            
            try:
                links = self.search_news_links(query, max_results=10)
                
                if not links:
                    logging.warning(f"未找到有效链接，跳过查询: {query}")
                    continue
                
                for link in links:
                    if total_saved >= min_total_news or saved_for_query >= max_news_per_query:
                        break
                    
                    if link in self.crawled_urls:
                        logging.debug(f"URL已被爬取过，跳过: {link}")
                        continue
                    
                    self.crawled_urls.add(link)
                    
                    metadata = self.extract_news_content(link)
                    if metadata and self.save_news(metadata, query):
                        total_saved += 1
                        saved_for_query += 1
                    
                    time.sleep(random.uniform(2, 4))
                
                logging.info(f"关键词 '{query}' 完成: 保存 {saved_for_query} 篇")
                time.sleep(random.uniform(5, 10))
                
            except Exception as e:
                logging.error(f"处理查询 '{query}' 时发生错误: {e}")
                continue

        # 保存已爬取URL列表
        self._save_crawled_urls()
        
        total_duration = (datetime.now() - start_time).total_seconds()
        
        stats = {
            'total_saved': total_saved,
            'total_duration': total_duration,
            'start_time': start_time.isoformat(),
            'end_time': datetime.now().isoformat()
        }
        
        logging.info(f"爬虫任务完成！总共保存 {total_saved} 篇新闻，用时 {total_duration:.1f}s")
        
        return stats
    
    def close_driver(self) -> None:
        """关闭会话"""
        if hasattr(self, 'session'):
            try:
                self.session.close()
                logging.info("会话已关闭")
            except Exception as e:
                logging.error(f"关闭会话时发生错误: {e}")
    
    def __enter__(self):
        """上下文管理器入口"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.close_driver()


def main(search_queries: list[str], output_dir: str):
    try:
        with NewsSpider(output_dir=output_dir, news_id_start=1) as spider:
            stats = spider.crawl_news(
                search_queries=search_queries, 
                max_news_per_query=4, 
                min_total_news=60
            )
            
            print(f"\n{'='*50}")
            print(f"爬取统计:")
            print(f"总共保存: {stats['total_saved']} 篇新闻")
            print(f"总用时: {stats['total_duration']:.1f} 秒")
            print(f"{'='*50}")
            
    except KeyboardInterrupt:
        logging.info("用户中断了爬虫程序")
    except Exception as e:
        logging.error(f"程序运行出错: {e}")


if __name__ == "__main__":
    search_queries = [
        # 科技封锁与新兴领域竞争
        "美国对华AI芯片出口管制 影响", "中国应对美国技术封锁对策", "美国生物安全法案 对中国药企影响",
        "中美人工智能治理 国际竞争", "量子计算领域 国际合作与限制",
        
        # 贸易摩擦与供应链重构
        "欧盟对中国电动汽车反补贴调查 最终措施", "美国对华加征关税 商品清单 2024 2025", 
        "全球供应链去风险化 对中国制造业影响", "中国稀土出口管制 国际反应", 
        "墨西哥作为中国对美出口中转站 风险",
        
        # 地缘政治与国际关系
        "2024美国大选后 对华政策实际影响", "俄乌冲突对中国经济及外交的长期影响 2025", 
        "中东局势紧张对中国能源安全的影响", "南海地区紧张局势 最新动态 2025", 
        "中欧关系 最新挑战与机遇 2025",
        
        # 金融制裁与投资限制
        "美国对华投资禁令 最新进展 2025", "中国企业应对美国金融制裁案例", 
        "美元结算体系风险与人民币国际化", "中概股在美国面临的审计监管挑战",
        
        # 内部经济与产业挑战
        "中国地方政府债务风险化解 2025", "房地产市场调整对中国经济的影响", 
        "中国粮食安全面临的挑战与对策", "中国人口结构变化对长期经济增长的影响",
    ]
    output_dir = "./news_data"
    main(search_queries, output_dir)
    print("爬虫任务已完成，结果保存在:", output_dir)