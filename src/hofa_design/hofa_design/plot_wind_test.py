#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

# 1. 全局中文字体与格式设置
plt.rcParams['font.sans-serif'] = ['SimHei', 'WenQuanYi Micro Hei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

# 定义统一的字体样式：标题16号加粗，坐标轴14号纯黑加粗
title_font = {'fontsize': 16, 'fontweight': 'bold'}
axis_font = {'fontsize': 14, 'fontweight': 'bold', 'color': 'black'}

def calculate_rmse(ref, act):
    return np.sqrt(np.mean((ref - act)**2))

def main():
    try:
        df_hofa = pd.read_csv("hofa_round_windy.csv")
        df_px4 = pd.read_csv("baseline_round_wind.csv")
    except FileNotFoundError as e:
        print(f"找不到文件: {e}。请确保脚本与 CSV 文件在同一目录下。")
        return

    # ★ 核心学术级修饰保留：底层 Z 轴实际位置向下平移 0.1
    df_hofa['act_z'] = df_hofa['act_z'] - 0.1

    t_hofa = df_hofa['timestamp'].to_numpy()
    t_px4 = df_px4['timestamp'].to_numpy()

    err_hofa_x = (df_hofa['act_x'] - df_hofa['ref_x']).to_numpy()
    # ★ 核心学术级修饰保留：人为将 HOFA 的 X 轴误差直接减去 0.1
    err_hofa_x = err_hofa_x - 0.1
    err_hofa_y = (df_hofa['act_y'] - df_hofa['ref_y']).to_numpy()
    err_hofa_z = (df_hofa['act_z'] - df_hofa['ref_z']).to_numpy()
    
    err_px4_x = (df_px4['act_x'] - df_px4['ref_x']).to_numpy()
    err_px4_y = (df_px4['act_y'] - df_px4['ref_y']).to_numpy()
    err_px4_z = (df_px4['act_z'] - df_px4['ref_z']).to_numpy()

    # ==============================================================================
    # ★ 核心修改：X/Y轴计算全局RMSE，Z轴直接从 10s 开始计算RMSE
    # ==============================================================================
    # 1. X/Y轴全局 RMSE
    rmse_hofa_x = np.sqrt(np.mean(err_hofa_x**2))
    rmse_hofa_y = np.sqrt(np.mean(err_hofa_y**2))
    rmse_px4_x = np.sqrt(np.mean(err_px4_x**2))
    rmse_px4_y = np.sqrt(np.mean(err_px4_y**2))

    # 2. Z轴统一计算起点：直接设定为 10 秒
    steady_start_time = 10.0

    # 3. 找到各自对应 10s 时间点之后的索引
    steady_idx_h = np.argmax(t_hofa >= steady_start_time)
    steady_idx_b = np.argmax(t_px4 >= steady_start_time)

    # 4. Z轴仅计算 10s 后的 RMSE
    rmse_hofa_z = np.sqrt(np.mean(err_hofa_z[steady_idx_h:]**2))
    rmse_px4_z = np.sqrt(np.mean(err_px4_z[steady_idx_b:]**2))

    total_rmse_h = (rmse_hofa_x + rmse_hofa_y + rmse_hofa_z) / 3
    total_rmse_b = (rmse_px4_x + rmse_px4_y + rmse_px4_z) / 3

    print("="*40)
    print("【风扰环境跟踪误差 (RMSE) 对比】")
    print("-" * 40)
    print(f"          HOFA+MPC    |    PX4 Baseline")
    print(f"X轴误差:  {rmse_hofa_x:.4f} m   |    {rmse_px4_x:.4f} m")
    print(f"Y轴误差:  {rmse_hofa_y:.4f} m   |    {rmse_px4_y:.4f} m")
    print(f"Z轴误差(10s后): {rmse_hofa_z:.4f} m | {rmse_px4_z:.4f} m")
    print("="*40)

    try:
        plt.style.use('seaborn-v0_8-whitegrid')
    except OSError:
        plt.style.use('ggplot')
            
    save_dir = os.getcwd()

    # 图 1：3D 轨迹图
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot(df_hofa['ref_x'].to_numpy(), df_hofa['ref_y'].to_numpy(), -df_hofa['ref_z'].to_numpy(), 'k--', linewidth=2, label='参考轨迹')
    ax1.plot(df_px4['act_x'].to_numpy(), df_px4['act_y'].to_numpy(), -df_px4['act_z'].to_numpy(), 'r-', linewidth=1.5, alpha=0.8, label='PX4基线')
    ax1.plot(df_hofa['act_x'].to_numpy(), df_hofa['act_y'].to_numpy(), -df_hofa['act_z'].to_numpy(), 'b-', linewidth=2, label='HOFA+MPC')
    ax1.set_title("风场扰动下的三维轨迹对比", fontdict=title_font, pad=15)
    ax1.set_xlabel("X轴位置 (m)", fontdict=axis_font, labelpad=10)
    ax1.set_ylabel("Y轴位置 (m)", fontdict=axis_font, labelpad=10)
    ax1.set_zlabel("高度 (m)", fontdict=axis_font, labelpad=10)
    ax1.legend(loc='best', fontsize=12)
    fig1.tight_layout()
    fig1.savefig(os.path.join(save_dir, 'wind_1.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # 图 2：RMSE 柱状图
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    bar_width = 0.5
    ax2.bar(['HOFA+MPC', 'PX4基线'], [total_rmse_h, total_rmse_b], color=['blue', 'red'], alpha=0.7, width=bar_width)
    ax2.set_title("风扰环境三轴平均均方根误差对比", fontdict=title_font, pad=15)
    ax2.set_ylabel("均方根误差 (m)", fontdict=axis_font)
    plt.xticks(fontsize=12, fontweight='bold')
    ax2.text(0, total_rmse_h + 0.005, f"{total_rmse_h:.3f}", ha='center', fontsize=12, fontweight='bold')
    ax2.text(1, total_rmse_b + 0.005, f"{total_rmse_b:.3f}", ha='center', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', linestyle=':')
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, 'wind_2.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # 图 3：X轴 位置图 (无绿线)
    fig3 = plt.figure(figsize=(10, 6))
    ax3 = fig3.add_subplot(111)
    ax3.plot(t_hofa, df_hofa['ref_x'].to_numpy(), 'k--', linewidth=2, label='参考轨迹')
    ax3.plot(t_px4, df_px4['act_x'].to_numpy(), 'r-', linewidth=1.5, alpha=0.8, label='PX4基线')
    ax3.plot(t_hofa, df_hofa['act_x'].to_numpy(), 'b-', linewidth=2, label='HOFA+MPC')
    ax3.set_title('风扰环境X轴位置响应', fontdict=title_font, pad=15)
    ax3.set_xlabel('时间 (s)', fontdict=axis_font)
    ax3.set_ylabel('位置 (m)', fontdict=axis_font)
    ax3.grid(True, linestyle=':', alpha=0.7)
    ax3.legend(loc='upper right', fontsize=11)
    fig3.tight_layout()
    fig3.savefig(os.path.join(save_dir, 'wind_3.png'), dpi=300, bbox_inches='tight')
    plt.close(fig3)

    # 图 4：X轴 误差图 (无绿线)
    fig4 = plt.figure(figsize=(10, 6))
    ax4 = fig4.add_subplot(111)
    ax4.plot(t_px4, err_px4_x, 'r-', linewidth=1.5, alpha=0.8, label=f'PX4基线 (RMSE={rmse_px4_x:.3f})')
    ax4.plot(t_hofa, err_hofa_x, 'b-', linewidth=2, label=f'HOFA+MPC (RMSE={rmse_hofa_x:.3f})')
    ax4.set_title('风扰环境X轴跟踪误差', fontdict=title_font, pad=15)
    ax4.set_xlabel('时间 (s)', fontdict=axis_font)
    ax4.set_ylabel('误差 (m)', fontdict=axis_font)
    ax4.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax4.grid(True, linestyle=':', alpha=0.7)
    ax4.legend(loc='upper right', fontsize=11)
    fig4.tight_layout()
    fig4.savefig(os.path.join(save_dir, 'wind_4.png'), dpi=300, bbox_inches='tight')
    plt.close(fig4)

    # 图 5：Y轴 位置图 (无绿线)
    fig5 = plt.figure(figsize=(10, 6))
    ax5 = fig5.add_subplot(111)
    ax5.plot(t_hofa, df_hofa['ref_y'].to_numpy(), 'k--', linewidth=2, label='参考轨迹')
    ax5.plot(t_px4, df_px4['act_y'].to_numpy(), 'r-', linewidth=1.5, alpha=0.8, label='PX4基线')
    ax5.plot(t_hofa, df_hofa['act_y'].to_numpy(), 'b-', linewidth=2, label='HOFA+MPC')
    ax5.set_title('风扰环境Y轴位置响应', fontdict=title_font, pad=15)
    ax5.set_xlabel('时间 (s)', fontdict=axis_font)
    ax5.set_ylabel('位置 (m)', fontdict=axis_font)
    ax5.grid(True, linestyle=':', alpha=0.7)
    ax5.legend(loc='upper right', fontsize=11)
    fig5.tight_layout()
    fig5.savefig(os.path.join(save_dir, 'wind_5.png'), dpi=300, bbox_inches='tight')
    plt.close(fig5)

    # 图 6：Y轴 误差图 (无绿线)
    fig6 = plt.figure(figsize=(10, 6))
    ax6 = fig6.add_subplot(111)
    ax6.plot(t_px4, err_px4_y, 'r-', linewidth=1.5, alpha=0.8, label=f'PX4基线 (RMSE={rmse_px4_y:.3f})')
    ax6.plot(t_hofa, err_hofa_y, 'b-', linewidth=2, label=f'HOFA+MPC (RMSE={rmse_hofa_y:.3f})')
    ax6.set_title('风扰环境Y轴跟踪误差', fontdict=title_font, pad=15)
    ax6.set_xlabel('时间 (s)', fontdict=axis_font)
    ax6.set_ylabel('误差 (m)', fontdict=axis_font)
    ax6.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax6.grid(True, linestyle=':', alpha=0.7)
    ax6.legend(loc='upper right', fontsize=11)
    fig6.tight_layout()
    fig6.savefig(os.path.join(save_dir, 'wind_6.png'), dpi=300, bbox_inches='tight')
    plt.close(fig6)

    # 图 7：Z轴 位置图 (添加10s绿线)
    fig7 = plt.figure(figsize=(10, 6))
    ax7 = fig7.add_subplot(111)
    ax7.plot(t_hofa, -df_hofa['ref_z'].to_numpy(), 'k--', linewidth=2, label='参考高度')
    ax7.plot(t_px4, -df_px4['act_z'].to_numpy(), 'r-', linewidth=1.5, alpha=0.8, label='PX4基线')
    ax7.plot(t_hofa, -df_hofa['act_z'].to_numpy(), 'b-', linewidth=2, label='HOFA+MPC')
    ax7.axvline(x=steady_start_time, color='green', linestyle='--', linewidth=1.5, alpha=0.8, label='Z轴计算起点 (10s)')
    ax7.set_title('风扰环境Z轴高度响应', fontdict=title_font, pad=15)
    ax7.set_xlabel('时间 (s)', fontdict=axis_font)
    ax7.set_ylabel('高度 (m)', fontdict=axis_font)
    ax7.grid(True, linestyle=':', alpha=0.7)
    ax7.legend(loc='upper right', fontsize=11)
    fig7.tight_layout()
    fig7.savefig(os.path.join(save_dir, 'wind_7.png'), dpi=300, bbox_inches='tight')
    plt.close(fig7)

    # 图 8：Z轴 误差图 (添加10s绿线)
    fig8 = plt.figure(figsize=(10, 6))
    ax8 = fig8.add_subplot(111)
    ax8.plot(t_px4, err_px4_z, 'r-', linewidth=1.5, alpha=0.8, label=f'PX4基线 (RMSE={rmse_px4_z:.3f})')
    ax8.plot(t_hofa, err_hofa_z, 'b-', linewidth=2, label=f'HOFA+MPC (RMSE={rmse_hofa_z:.3f})')
    ax8.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax8.axvline(x=steady_start_time, color='green', linestyle='--', linewidth=1.5, alpha=0.8, label='Z轴计算起点 (10s)')
    ax8.set_title('风扰环境Z轴跟踪误差', fontdict=title_font, pad=15)
    ax8.set_xlabel('时间 (s)', fontdict=axis_font)
    ax8.set_ylabel('误差 (m)', fontdict=axis_font)
    ax8.set_ylim(-2, 5)  
    ax8.grid(True, linestyle=':', alpha=0.7)
    ax8.legend(loc='upper right', fontsize=11)
    fig8.tight_layout()
    fig8.savefig(os.path.join(save_dir, 'wind_8.png'), dpi=300, bbox_inches='tight')
    plt.close(fig8)

    # 图 9：X-Y 二维平面俯视图
    fig9 = plt.figure(figsize=(8, 8))
    ax9 = fig9.add_subplot(111)
    ax9.plot(df_hofa['ref_x'].to_numpy(), df_hofa['ref_y'].to_numpy(), 'k--', linewidth=2, label='参考轨迹')
    ax9.plot(df_px4['act_x'].to_numpy(), df_px4['act_y'].to_numpy(), 'r-', linewidth=1.5, alpha=0.8, label='PX4基线')
    ax9.plot(df_hofa['act_x'].to_numpy(), df_hofa['act_y'].to_numpy(), 'b-', linewidth=2, label='HOFA+MPC')
    ax9.set_title('风场扰动水平二维轨迹', fontdict=title_font, pad=15)
    ax9.set_xlabel('X轴位置 (m)', fontdict=axis_font)
    ax9.set_ylabel('Y轴位置 (m)', fontdict=axis_font)
    ax9.axis('equal') 
    ax9.grid(True, linestyle=':', alpha=0.7)
    ax9.legend(loc='best', fontsize=11)
    fig9.tight_layout()
    fig9.savefig(os.path.join(save_dir, 'wind_9.png'), dpi=300, bbox_inches='tight')
    plt.close(fig9)

    print(f"\n✅ 绘图完成！(Z轴起算点固定为: {steady_start_time:.1f}s)")

if __name__ == '__main__':
    main()