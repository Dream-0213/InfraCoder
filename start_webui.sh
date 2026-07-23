#!/bin/bash
# InfraCoder Web UI 启动脚本
# 用法： ./start_webui.sh [stop|restart|status]

cd /home/ubuntu/XYP/InfraCoder || exit 1
source venv/bin/activate

PIDFILE=/tmp/infracoder_webui.pid
LOGFILE=webui.log

case "${1:-start}" in
  start)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "Web UI 已在运行，PID=$(cat "$PIDFILE")"
      exit 0
    fi
    echo "启动 InfraCoder Web UI ..."
    nohup python3 webui.py > "$LOGFILE" 2>&1 &
    echo $! > "$PIDFILE"
    echo "PID: $!"
    echo "访问地址: http://192.168.15.119:7860"
    echo "日志文件: $LOGFILE"
    ;;
  stop)
    if [ -f "$PIDFILE" ]; then
      kill "$(cat "$PIDFILE")" 2>/dev/null && echo "已停止" || echo "进程不存在"
      rm -f "$PIDFILE"
    else
      echo "未在运行"
    fi
    ;;
  restart)
    "$0" stop
    sleep 1
    "$0" start
    ;;
  status)
    if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
      echo "Web UI 运行中，PID=$(cat "$PIDFILE")"
    else
      echo "Web UI 未运行"
    fi
    ;;
  *)
    echo "用法: $0 {start|stop|restart|status}"
    exit 1
    ;;
esac
