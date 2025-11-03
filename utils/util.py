import os, sys, contextlib, logging
from datetime import datetime


@contextlib.contextmanager
def redirect_output_to_file(file_path):
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    f = open(file_path, "a", encoding="utf-8")
    # f = open(file_path, "w", encoding="utf-8")

    # --- 打印分隔符（含时间戳） ---
    f.write("\n" + "=" * 80 + "\n")
    f.write(f"[LOG START] {datetime.now():%Y-%m-%d %H:%M:%S}\n")
    f.write("=" * 80 + "\n")
    f.flush()

    # 保存旧的 sys.stdout/sys.stderr
    old_stdout, old_stderr = sys.stdout, sys.stderr
    sys.stdout = sys.stderr = f

    # 保存并替换 logging handler 的 stream
    old_streams = []
    for handler in logging.root.handlers:
        if isinstance(handler, logging.StreamHandler):
            old_streams.append((handler, handler.stream))
            handler.stream = f

    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()

        # 恢复 logging handler
        for handler, old_stream in old_streams:
            handler.stream = old_stream

        sys.stdout, sys.stderr = old_stdout, old_stderr
        f.close()
