import os
from pathlib import Path

def download_url(url: str, save_path: str):
    """
    下载URL并保存到指定路径
    
    Args:
        url (str): 要下载的URL
        save_path (str): 保存路径
    """
    import requests
    import tqdm
    
    # 确保保存目录存在
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    
    response = requests.get(url, stream=True)
    response.raise_for_status()  # 检查HTTP错误
    
    total_size = int(response.headers.get('content-length', 0))
    
    with open(save_path, 'wb') as f:
        with tqdm.tqdm(total=total_size, unit='B', unit_scale=True, desc=os.path.basename(save_path)) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    pbar.update(len(chunk))

def unix_to_windows_path(unix_path: str, drive_map: dict = None) -> str:

    """
    将 Unix/Cygwin 风格的路径转换为 Windows 风格的路径。

    参数:
        unix_path (str): Unix/Cygwin 风格的路径（例如 `/d/project/file` 或 `/cygdrive/c/Users`）
        drive_map (dict): 自定义盘符映射（例如 `{'/c/': 'C:\\', '/d/': 'E:\\'}`），默认自动映射 `/x/` → `X:\`

    返回:
        str: Windows 风格的路径（例如 `D:\project\file`）
    """
    # 默认盘符映射（/x/ → X:\）
    default_drive_map = {f'/{d}/': f'{d.upper()}:\\' for d in 'abcdefghijklmnopqrstuvwxyz'}
    drive_map = drive_map or default_drive_map

    # 处理 Cygwin 的 /cygdrive/x/ 格式
    if unix_path.startswith('/cygdrive/'):
        parts = unix_path.split('/')
        drive_letter = parts[2].lower()
        unix_path = f'/{drive_letter}/' + '/'.join(parts[3:])

    # 替换盘符（例如 /d/ → D:\）
    for unix_drive, win_drive in drive_map.items():
        if unix_path.startswith(unix_drive):
            unix_path = unix_path.replace(unix_drive, win_drive, 1)
            break

    # 统一转换为 Windows 路径分隔符
    windows_path = unix_path.replace('/', '\\')

    # 使用 pathlib 规范化路径（解决 `.`、`..`、多余分隔符等问题）
    windows_path = str(Path(windows_path).resolve())

    return windows_path


