"""Per-user configuration - personalization for multi-user setups.

每个用户有自己的配置文件，覆盖全局默认设置。
配置文件路径：~/.infracoder/users/<username>.yaml
"""

from __future__ import annotations

import os
from pathlib import Path

try:
    import yaml
except ImportError:
    yaml = None


class UserConfig:
    """用户个性化配置。"""

    @staticmethod
    def _get_project_dir() -> Path:
        """优先使用项目目录下的 .infracoder，找不到则用 home 目录。"""
        cwd_dir = Path.cwd().resolve()
        # 从 cwd 向上查找 .infracoder 目录
        for p in [cwd_dir] + list(cwd_dir.parents):
            if (p / ".infracoder").exists():
                return p
        # 找不到时用 home 目录
        return Path.home()

    def __init__(self, username: str | None = None):
        self.username = username or self._detect_user()
        self.config_dir = self._get_project_dir() / ".infracoder" / "users"
        self.config_dir.mkdir(parents=True, exist_ok=True)
        self.config_path = self.config_dir / f"{self.username}.yaml"

        self.preferred_model: str | None = None
        self.output_style: str = "default"       # default, concise, detailed, bullet
        self.disabled_tools: list[str] = []       # 禁用的工具名列表
        self.preferred_language: str = "chinese"  # chinese, english
        self.session_tags: list[str] = []

        self._load()

    @staticmethod
    def _detect_user() -> str:
        """检测当前系统用户名。"""
        return os.environ.get("USER") or os.environ.get("USERNAME") or "default"

    def _load(self):
        """从 YAML 文件加载配置。"""
        if not self.config_path.exists():
            return

        if yaml is None:
            return

        try:
            data = yaml.safe_load(self.config_path.read_text(encoding="utf-8"))
            if not data:
                return
            self.preferred_model = data.get("preferred_model") or self.preferred_model
            self.output_style = data.get("output_style", self.output_style)
            self.disabled_tools = data.get("disabled_tools", self.disabled_tools)
            self.preferred_language = data.get("preferred_language", self.preferred_language)
            self.session_tags = data.get("session_tags", self.session_tags)
        except Exception:
            pass

    def save(self):
        """保存配置到 YAML 文件。"""
        if yaml is None:
            return

        data = {
            "preferred_model": self.preferred_model,
            "output_style": self.output_style,
            "disabled_tools": self.disabled_tools,
            "preferred_language": self.preferred_language,
            "session_tags": self.session_tags,
        }
        self.config_path.write_text(
            yaml.safe_dump(data, allow_unicode=True, default_flow_style=False),
            encoding="utf-8",
        )

    def add_tag(self, tag: str):
        """给当前用户添加一个会话标签。"""
        if tag not in self.session_tags:
            self.session_tags.append(tag)
            self.save()

    def remove_tag(self, tag: str):
        """移除一个标签。"""
        if tag in self.session_tags:
            self.session_tags.remove(tag)
            self.save()

    def is_tool_disabled(self, tool_name: str) -> bool:
        """检查某个工具是否被该用户禁用。"""
        return tool_name in self.disabled_tools

    def describe(self) -> str:
        """返回配置的友好描述。"""
        parts = [f"User: {self.username}"]
        if self.preferred_model:
            parts.append(f"Model: {self.preferred_model}")
        parts.append(f"Style: {self.output_style}")
        parts.append(f"Language: {self.preferred_language}")
        if self.disabled_tools:
            parts.append(f"Disabled tools: {', '.join(self.disabled_tools)}")
        if self.session_tags:
            parts.append(f"Tags: {', '.join(self.session_tags)}")
        return " | ".join(parts)


# 简单的个人资料模板生成
USER_CONFIG_TEMPLATE = """# {username} 的 InfraCoder 配置
# 去掉可选字段前面的 # 来启用

# preferred_model: deepseek-chat
# output_style: concise       # default, concise, detailed, bullet
# preferred_language: chinese  # chinese, english
# disabled_tools:
#   - bash
#   - workflow
"""


def init_user_config(username: str | None = None) -> UserConfig:
    """初始化用户配置，如不存在则创建模板。"""
    username = username or UserConfig._detect_user()
    config = UserConfig(username)

    # 如配置文件不存在，生成模板
    if not config.config_path.exists():
        config.config_path.write_text(
            USER_CONFIG_TEMPLATE.format(username=username),
            encoding="utf-8",
        )

    return config
