#!/bin/bash

# 显示文件状态的函数
show_status() {
    echo "=== 已读文件 ==="
    for file in *.txt; do
        if attr -g "status" "$file" &>/dev/null; then
            echo "✓ $file"
        fi
    done

    echo -e "\n=== 未读文件 ==="
    for file in *.txt; do
        if ! attr -g "status" "$file" &>/dev/null; then
            # 创建可点击的链接，使用 OSC 8 转义序列
            printf '\e]8;;cursor://file/%s\e\\○ %s\e]8;;\e\\\n' "$(realpath "$file")" "$file"
        fi
    done
}

# 主菜单
while true; do
    echo -e "\n请选择操作："
    echo "1. 显示文件状态"
    echo "2. 标记文件为已读"
    echo "3. 退出"
    read -p "输入选项 (1-3): " choice

    case $choice in
        1)
            show_status
            ;;
        2)
            echo -e "\n=== 未读文件 ==="
            unread_files=()
            i=1
            for file in *.txt; do
                if ! attr -g "status" "$file" &>/dev/null; then
                    unread_files+=("$file")
                    # 创建带编号的可点击链接
                    printf '%d. \e]8;;cursor://file/%s\e\\○ %s\e]8;;\e\\\n' "$i" "$(realpath "$file")" "$file"
                    ((i++))
                fi
            done
            
            if [ ${#unread_files[@]} -eq 0 ]; then
                echo "没有未读文件"
                continue
            fi
            
            read -p "选择要标记为已读的文件编号 (多个编号用空格分隔，输入 'q' 返回，输入 'a' 选择全部): " input
            if [ "$input" = "q" ]; then
                continue
            fi
            
            if [ "$input" = "a" ]; then
                # 标记所有文件为已读
                for file in "${unread_files[@]}"; do
                    attr -s status -V "read" "$file"
                    echo "$file 已标记为已读"
                done
            else
                # 处理多个编号输入
                for num in $input; do
                    if [ "$num" -ge 1 ] && [ "$num" -le ${#unread_files[@]} ]; then
                        attr -s status -V "read" "${unread_files[$num-1]}"
                        echo "${unread_files[$num-1]} 已标记为已读"
                    else
                        echo "无效的选择: $num"
                    fi
                done
            fi
            ;;
        3)
            echo "退出程序"
            exit 0
            ;;
        *)
            echo "无效的选择，请重试"
            ;;
    esac
done