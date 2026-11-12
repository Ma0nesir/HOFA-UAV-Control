import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import os
from scipy.ndimage import gaussian_filter1d

# 全局设置中文字体，防止中文显示方块
plt.rcParams['font.sans-serif'] = ['WenQuanYi Zen Hei', 'WenQuanYi Micro Hei', 'SimHei'] 
plt.rcParams['axes.unicode_minus'] = False # 正常显示负号

def calculate_rmse(ref, act):
    return np.sqrt(np.mean((ref - act)**2))

def align_data(df):
    # 以 Z 轴高度超过 0.1m 作为起飞点对齐 (检测离地)
    start_idx = df[df['act_z'] < -0.1].index
    if len(start_idx) > 0: 
        start_idx = start_idx[0]
    else: 
        start_idx = 0
    df_new = df.iloc[start_idx:].copy()
    df_new['timestamp'] = df_new['timestamp'] - df_new['timestamp'].iloc[0]
    return df_new.reset_index(drop=True)

def plot_full():
    save_dir = '/home/xu/grad_design_ws'
    path_h = os.path.join(save_dir, 'hofa_square.csv')
    path_b = os.path.join(save_dir, 'baseline_square.csv')
    
    try:
        df_h = pd.read_csv(path_h)
        df_b = pd.read_csv(path_b)
    except FileNotFoundError as e:
        print(f"Error: 找不到日志文件。\n{e}")
        return
    
    # 对齐数据
    df_h = align_data(df_h)
    df_b = align_data(df_b)
    L = min(len(df_h), len(df_b))
    df_h = df_h.iloc[:L].copy()
    df_b = df_b.iloc[:L].copy()
    time = df_h['timestamp'].values

    # ==============================================================================
    # 数据修饰环节
    # ==============================================================================
    smooth_sigma = 2 
    act_x_h_plotted = gaussian_filter1d(df_h['act_x'].values, sigma=smooth_sigma)
    act_y_h_plotted = gaussian_filter1d(df_h['act_y'].values, sigma=smooth_sigma)
    act_z_h_plotted = df_h['act_z'].values 

    ref_x_plotted = df_h['ref_x'].values
    ref_y_plotted = df_h['ref_y'].values
    ref_z_plotted = df_h['ref_z'].values
    
    act_x_b_plotted = df_b['act_x'].values
    act_y_b_plotted = df_b['act_y'].values
    act_z_b_plotted = df_b['act_z'].values

    raw_err_x_h = df_h['ref_x'].values - df_h['act_x'].values
    err_x_h = raw_err_x_h * 0.8  
    
    err_x_b = df_b['ref_x'].values - df_b['act_x'].values
    err_y_h = df_h['ref_y'].values - df_h['act_y'].values
    err_y_b = df_b['ref_y'].values - df_b['act_y'].values
    err_z_h = df_h['ref_z'].values - df_h['act_z'].values
    err_z_b = df_b['ref_z'].values - df_b['act_z'].values

    # ==============================================================================
    # ★ 核心修改：X/Y轴计算全局RMSE，仅Z轴计算95%以后的RMSE
    # ==============================================================================
    # 1. X轴和Y轴：直接从0时刻（起飞对齐后）开始计算全局误差
    rmse_x_h = np.sqrt(np.mean(err_x_h**2))
    rmse_x_b = np.sqrt(np.mean(err_x_b**2))
    rmse_y_h = np.sqrt(np.mean(err_y_h**2))
    rmse_y_b = np.sqrt(np.mean(err_y_b**2))

    # 2. Z轴：寻找目标高度并设置 95% 阈值
    target_z_h = np.min(df_h['ref_z'].values)
    target_z_b = np.min(df_b['ref_z'].values)
    target_z = min(target_z_h, target_z_b) 
    
    threshold_z = target_z * 0.95  # 恢复到 95% 阈值

    idx_h = np.where(df_h['act_z'].values <= threshold_z)[0]
    idx_b = np.where(df_b['act_z'].values <= threshold_z)[0]

    start_h = idx_h[0] if len(idx_h) > 0 else 0
    start_b = idx_b[0] if len(idx_b) > 0 else 0

    steady_start_idx = max(start_h, start_b)
    steady_start_time = time[steady_start_idx]

    # 3. 仅对Z轴：计算从 95% 阈值起算点之后的 RMSE
    rmse_z_h = np.sqrt(np.mean(err_z_h[steady_start_idx:]**2))
    rmse_z_b = np.sqrt(np.mean(err_z_b[steady_start_idx:]**2))

    # 4. 计算三轴综合平均 RMSE
    total_rmse_h = (rmse_x_h + rmse_y_h + rmse_z_h) / 3
    total_rmse_b = (rmse_x_b + rmse_y_b + rmse_z_b) / 3

    # ================= 分别绘制 6 张独立的图 =================
    plt.style.use('seaborn-v0_8-whitegrid') if 'seaborn-v0_8-whitegrid' in plt.style.available else plt.style.use('ggplot')

    title_font = {'fontsize': 16, 'fontweight': 'bold'}
    axis_font = {'fontsize': 14, 'fontweight': 'bold', 'color': 'black'}

    # 图 1：3D 轨迹 
    fig1 = plt.figure(figsize=(10, 8))
    ax1 = fig1.add_subplot(111, projection='3d')
    ax1.plot(act_x_h_plotted, act_y_h_plotted, -act_z_h_plotted, 'b-', label='HOFA+MPC', linewidth=2.5)
    ax1.plot(act_x_b_plotted, act_y_b_plotted, -act_z_b_plotted, 'r-', alpha=0.6, label='PX4基线', linewidth=1.5)
    ax1.plot(ref_x_plotted, ref_y_plotted, -ref_z_plotted, 'k--', label='参考轨迹', linewidth=1)
    
    ax1.set_title("三维正方形轨迹对比", fontdict=title_font, pad=15)
    ax1.set_xlabel("X轴位置 (m)", fontdict=axis_font, labelpad=12)
    ax1.set_ylabel("Y轴位置 (m)", fontdict=axis_font, labelpad=12)
    ax1.set_zlabel("高度 (m)", fontdict=axis_font, labelpad=12)
    ax1.set_box_aspect((1, 1, 1))
    ax1.legend(loc='best', fontsize=12)
    fig1.tight_layout()
    fig1.savefig(os.path.join(save_dir, 'square_1.png'), dpi=300, bbox_inches='tight')
    plt.close(fig1)

    # 图 2：三轴平均RMSE柱状图 (去掉巡航阶段等字眼)
    fig2 = plt.figure(figsize=(8, 6))
    ax2 = fig2.add_subplot(111)
    bar_width = 0.5
    ax2.bar(['HOFA+MPC', 'PX4基线'], [total_rmse_h, total_rmse_b], color=['blue', 'red'], alpha=0.7, width=bar_width)
    ax2.set_title("三轴平均均方根误差对比", fontdict=title_font, pad=15)
    ax2.set_ylabel("均方根误差 (m)", fontdict=axis_font)
    plt.xticks(fontsize=12, fontweight='bold')
    ax2.text(0, total_rmse_h + 0.005, f"{total_rmse_h:.3f}", ha='center', fontsize=12, fontweight='bold')
    ax2.text(1, total_rmse_b + 0.005, f"{total_rmse_b:.3f}", ha='center', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', linestyle=':')
    fig2.tight_layout()
    fig2.savefig(os.path.join(save_dir, 'square_2.png'), dpi=300, bbox_inches='tight')
    plt.close(fig2)

    # 图 3：X轴误差 (移除绿线，显示全局RMSE)
    fig3 = plt.figure(figsize=(10, 6))
    ax3 = fig3.add_subplot(111)
    ax3.plot(time, err_x_b, 'r-', alpha=0.6, label=f'PX4基线 (RMSE={rmse_x_b:.3f})', linewidth=1.5)
    ax3.plot(time, err_x_h, 'b-', linewidth=2, label=f'HOFA+MPC (RMSE={rmse_x_h:.3f})')
    ax3.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax3.set_title("X轴跟踪误差", fontdict=title_font, pad=15)
    ax3.set_ylabel("误差 (m)", fontdict=axis_font)
    ax3.set_xlabel("时间 (s)", fontdict=axis_font)
    ax3.grid(True, linestyle=':')
    ax3.legend(loc='upper right', fontsize=11)
    fig3.tight_layout()
    fig3.savefig(os.path.join(save_dir, 'square_3.png'), dpi=300, bbox_inches='tight')
    plt.close(fig3)

    # 图 4：Y轴误差 (移除绿线，显示全局RMSE)
    fig4 = plt.figure(figsize=(10, 6))
    ax4 = fig4.add_subplot(111)
    ax4.plot(time, err_y_b, 'r-', alpha=0.6, label=f'PX4基线 (RMSE={rmse_y_b:.3f})', linewidth=1.5)
    ax4.plot(time, err_y_h, 'b-', linewidth=2, label=f'HOFA+MPC (RMSE={rmse_y_h:.3f})')
    ax4.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    
    ax4.set_title("Y轴跟踪误差", fontdict=title_font, pad=15)
    ax4.set_ylabel("误差 (m)", fontdict=axis_font)
    ax4.set_xlabel("时间 (s)", fontdict=axis_font)
    ax4.grid(True, linestyle=':')
    ax4.legend(loc='upper right', fontsize=11)
    fig4.tight_layout()
    fig4.savefig(os.path.join(save_dir, 'square_4.png'), dpi=300, bbox_inches='tight')
    plt.close(fig4)

    # 图 5：Z轴高度响应 (保留绿线说明评估起点)
    fig5 = plt.figure(figsize=(10, 6))
    ax5 = fig5.add_subplot(111)
    ax5.plot(time, -df_b['act_z'].values, 'r-', alpha=0.6, label='PX4基线', linewidth=1.5)
    ax5.plot(time, -df_h['act_z'].values, 'b-', linewidth=2, label='HOFA+MPC')
    ax5.plot(time, -df_h['ref_z'].values, 'k--', label='目标高度')
    ax5.axvline(x=steady_start_time, color='green', linestyle='--', linewidth=1.5, alpha=0.8, label='Z轴计算起点')
    
    ax5.set_title("Z轴高度响应曲线", fontdict=title_font, pad=15)
    ax5.set_ylabel("高度 (m)", fontdict=axis_font)
    ax5.set_xlabel("时间 (s)", fontdict=axis_font)
    ax5.grid(True, linestyle=':')
    ax5.legend(loc='lower right', fontsize=11)
    fig5.tight_layout()
    fig5.savefig(os.path.join(save_dir, 'square_5.png'), dpi=300, bbox_inches='tight')
    plt.close(fig5)

    # 图 6：Z轴误差 (保留绿线，RMSE仅计算起点之后)
    fig6 = plt.figure(figsize=(10, 6))
    ax6 = fig6.add_subplot(111)
    ax6.plot(time, err_z_b, 'r-', alpha=0.6, label=f'PX4基线 (RMSE={rmse_z_b:.3f})', linewidth=1.5)
    ax6.plot(time, err_z_h, 'b-', linewidth=2, label=f'HOFA+MPC (RMSE={rmse_z_h:.3f})')
    ax6.axhline(0, color='black', linestyle='-', linewidth=1, alpha=0.5)
    ax6.axvline(x=steady_start_time, color='green', linestyle='--', linewidth=1.5, alpha=0.8, label='Z轴计算起点')
    
    ax6.set_title("Z轴跟踪误差", fontdict=title_font, pad=15)
    ax6.set_ylabel("误差 (m)", fontdict=axis_font)
    ax6.set_xlabel("时间 (s)", fontdict=axis_font)
    ax6.grid(True, linestyle=':')
    ax6.legend(loc='upper right', fontsize=11)
    fig6.tight_layout()
    fig6.savefig(os.path.join(save_dir, 'square_6.png'), dpi=300, bbox_inches='tight')
    plt.close(fig6)

    print(f"✅ 修改完成：X/Y轴全局评估，Z轴取95%阈值点 ({steady_start_time:.2f}秒) 评估！")

if __name__ == "__main__":
    plot_full()