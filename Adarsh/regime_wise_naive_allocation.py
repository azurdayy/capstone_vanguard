##This is a relative performance based implementation that reads various monthly regime models
##then it takes the regime wise outperformance to over/under allocate assets

import pandas as pd
import numpy as np
import datetime as dt
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from matplotlib.patches import Rectangle
import os
import warnings
warnings.filterwarnings("ignore")


def load_regime_data(regimename):
    if regimename=="kmeans":
        Regime_clusters=pd.read_csv("data/kmeans_regimes_k5.csv")
    elif regimename=="gmm":
        Regime_clusters_test=pd.read_csv("data/BGMM_Regime_Test.csv")
        Regime_clusters_train=pd.read_csv("data/BGMM_Regime_Train.csv")
        Regime_clusters = pd.concat([Regime_clusters_test, Regime_clusters_train])
    else:
        Regime_clusters=pd.read_csv("data/1_HMM_regime.csv")

    Regime_clusters.columns=["Date", "Regime"]
    Regime_clusters["Date"]=pd.to_datetime(Regime_clusters["Date"]).dt.to_period("M")
    Regime_clusters = Regime_clusters.sort_values(by="Date")
    monthly_ret["equityrp"]=monthly_ret["Russell 1000"]-monthly_ret["US Short-term Treasury"]
    monthly_ret["val-grwth"]=monthly_ret["Russell 1000 Value"]-monthly_ret["Russell 1000 Growth"]
    monthly_ret["creditprem"]=monthly_ret["US HY Corporate Bond"]-monthly_ret["US IG Corporate Bond"]
    monthly_ret["termprem"]=monthly_ret["US Long-term Treasury"]-monthly_ret["US Short-term Treasury"]
    monthly_ret["Sizeprem"]=monthly_ret["Russell 2000"]-monthly_ret["Russell 1000"]

    df = monthly_ret.merge(Regime_clusters, on="Date", how="left")
    df = df.sort_values("Date")

    return df

def get_regime_perf(df, forwardlook=12):

    cols_to_drop = [
        'Russell 1000', 'Russell 1000 Value', 'Russell 1000 Growth',
        'MSCI World ex USA Index (DM ex-US Equities)',
        'MSCI EM Index (EM Equities)', 'US Agg Bond', 'US Short-term Treasury',
        'US Long-term Treasury', 'US IG Corporate Bond', 'US HY Corporate Bond',
        'Global Agg ex-US Bond USD Hedged', 'Gold', 'Dollar', "RTY Index", "Russell 2000"
    ]

    df = df.drop(columns=cols_to_drop)
    asset_cols = df.columns.difference(["Date", "Regime"])

    # Calculate 3-month forward compound returns for each asset, and align it to that month, because we are aligned to month start
    for col in asset_cols:
        df[f"{col}_fwd"] = (1 + df[col]).rolling(forwardlook).apply(lambda x: x.prod() - 1).shift(-forwardlook)

    # Get the new forward return column names
    fwd_cols = [f"{col}_fwd" for col in asset_cols]

    df_pre = df

    # Calculate average forward returns by Regime for each period
    # Pre-2020
    result_fwd_pre = df_pre.groupby("Regime")[fwd_cols].mean()
    result_fwd_pre.columns = [col.replace("_fwd", "") for col in result_fwd_pre.columns]

    return result_fwd_pre

def get_signal_based_target_v2(base, signals, total_tilt_budget=0.20, 
                                signal_threshold=0.0):
    """
    Generate target weights based on factor signals with magnitude-aware budgeting.
    
    Key improvements:
    - Budget allocation reflects both signal strength AND quality (magnitude)
    - Weak signals (near zero) get minimal budget
    - Strong signals (large magnitude) get more budget
    
    Parameters:
    - base: Base allocation weights
    - signals: Dict with keys: equityrp, val-grwth, creditprem, termprem, Sizeprem
    - total_tilt_budget: Total percentage points available for tilting
    - signal_threshold: Minimum absolute signal value to act on (default 0.0)
    """
    target = base.copy()
    
    signal_names = ['equityrp', 'val-grwth', 'creditprem', 'termprem', 'Sizeprem']
    signal_values = np.array([signals[s] for s in signal_names])
    
    # Use absolute values for budget allocation (stronger signals get more budget)
    abs_signals = np.abs(signal_values)
    
    # Zero out signals below threshold
    abs_signals[abs_signals < signal_threshold] = 0
    
    # Allocate budget proportional to absolute strength
    if abs_signals.sum() > 0:
        signal_weights = abs_signals / abs_signals.sum()
    else:
        # No strong signals - return base weights unchanged
        return target
    
    # Allocate budget proportionally to signal strength
    budgets = total_tilt_budget * signal_weights
    
    # Apply tilts with direction based on signal sign
    # Equity Risk Premium tilt
    if abs_signals[0] > signal_threshold:
        direction = np.sign(signal_values[0])
        target["Russell 1000 Value"] += direction * budgets[0] * 0.4
        target["Russell 1000 Growth"] += direction * budgets[0] * 0.4
        target["Russell 2000"] += direction * budgets[0] * 0.2
    
    # Value vs Growth tilt
    if abs_signals[1] > signal_threshold:
        direction = np.sign(signal_values[1])
        target["Russell 1000 Value"] += direction * budgets[1]
        target["Russell 1000 Growth"] -= direction * budgets[1]
    
    # Credit spread tilt
    if abs_signals[2] > signal_threshold:
        direction = np.sign(signal_values[2])
        target["US HY Corporate Bond"] += direction * budgets[2]
        target["US IG Corporate Bond"] -= direction * budgets[2]
    
    # Term premium tilt
    if abs_signals[3] > signal_threshold:
        direction = np.sign(signal_values[3])
        target["US Long-term Treasury"] += direction * budgets[3]
        target["US Short-term Treasury"] -= direction * budgets[3]
    
    # Size premium tilt
    if abs_signals[4] > signal_threshold:
        direction = np.sign(signal_values[4])
        target["Russell 2000"] += direction * budgets[4]
        target["Russell 1000 Value"] -= direction * budgets[4] * 0.5
        target["Russell 1000 Growth"] -= direction * budgets[4] * 0.5
    
    return target


def optimize_weights_v2(target, base, current_weights=None, 
                        min_equity=0.50, max_equity=0.70, 
                        max_asset_deviation=0.10, 
                        turnover_penalty=0.001):
    """
    Optimized weight allocation with better-structured objective.
    
    Improvements:
    - Separate turnover penalty from deviation penalty
    - Option to penalize changes from current holdings (not just base)
    - Clearer constraint structure
    
    Parameters:
    - target: Target weights from signals
    - base: Strategic base weights
    - current_weights: Current portfolio weights (for turnover penalty)
    - min_equity, max_equity: Equity allocation bounds
    - max_asset_deviation: Max deviation per asset from base
    - turnover_penalty: Penalty for portfolio turnover
    """
    n_assets = len(base)
    equity_indices = [0, 1, 2]  # R1000V, R1000G, R2000
    
    if current_weights is None:
        current_weights = base
    
    def objective(w):
        # Primary: match target weights
        target_tracking_error = np.sum((w - target.values)**2)
        
        # Secondary: minimize distance from base (risk control)
        base_deviation = 0.3 * np.sum((w - base.values)**2)
        
        # Tertiary: minimize turnover (transaction costs)
        turnover = turnover_penalty * np.sum(np.abs(w - current_weights.values))
        
        return target_tracking_error + base_deviation + turnover
    
    # Constraints
    constraints = [
        {'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0},  # weights sum to 1
        {'type': 'ineq', 'fun': lambda w: np.sum(w[equity_indices]) - min_equity},  # equity >= min
        {'type': 'ineq', 'fun': lambda w: max_equity - np.sum(w[equity_indices])},  # equity <= max
        {'type': 'ineq', 'fun': lambda w: w}  # all weights >= 0 (no shorts)
    ]
    
    # Bounds: respect max deviation from base
    bounds = [
        (max(0, base.iloc[i] - max_asset_deviation), 
         min(1.0, base.iloc[i] + max_asset_deviation)) 
        for i in range(n_assets)
    ]
    
    # Initial guess: start from current weights
    x0 = current_weights.values
    
    result = minimize(
        objective, x0, 
        method='SLSQP', 
        bounds=bounds, 
        constraints=constraints,
        options={'ftol': 1e-9, 'maxiter': 1000}
    )
    
    if result.success:
        return pd.Series(result.x, index=base.index)
    else:
        print(f"Optimization failed: {result.message}")
        # Fallback: project target onto feasible region
        w = np.clip(target.values, [b[0] for b in bounds], [b[1] for b in bounds])
        w = np.maximum(w, 0)  # No shorts
        
        # Adjust equity to bounds
        equity_sum = w[equity_indices].sum()
        if equity_sum < min_equity:
            w[equity_indices] *= min_equity / equity_sum
        elif equity_sum > max_equity:
            w[equity_indices] *= max_equity / equity_sum
        
        w = w / w.sum()  # Normalize
        return pd.Series(w, index=base.index)


def generate_dynamic_weights_v2(result_fwd_pre, base, 
                                min_equity=0.50, max_equity=0.70,
                                total_tilt_budget=0.20,
                                signal_threshold=0.01,
                                turnover_penalty=0):
    """
    Generate dynamic weights with improved signal processing and optimization.
    
    Parameters:
    - result_fwd_pre: DataFrame with factor premium forecasts by regime
    - base: Series with base asset allocation
    - min_equity, max_equity: Total equity allocation bounds
    - total_tilt_budget: Total budget for tilting away from base
    - signal_threshold: Minimum signal strength to act on (e.g., 1% = 0.01)
    - turnover_penalty: Weight on turnover minimization
    """
    weights_df = pd.DataFrame(index=result_fwd_pre.index, columns=base.index)
    
    # Track previous weights for turnover penalty
    previous_weights = base.copy()
    
    for i, row in result_fwd_pre.iterrows():
        signals = {
            'equityrp': row["equityrp"],
            'val-grwth': row["val-grwth"],
            'creditprem': row["creditprem"],
            'termprem': row["termprem"],
            'Sizeprem': row["Sizeprem"]
        }
        
        # Get target weights from signals (magnitude-aware)
        target = get_signal_based_target_v2(
            base, signals, total_tilt_budget, signal_threshold
        )
        
        # Optimize weights with turnover penalty
        w = optimize_weights_v2(
            target, base, 
            current_weights=previous_weights,
            min_equity=min_equity, 
            max_equity=max_equity,
            turnover_penalty=turnover_penalty
        )
        
        weights_df.loc[i] = w
        previous_weights = w  # Update for next iteration
    
    weights_df = weights_df.astype(float)
    
    weights_df = weights_df.add_prefix("Weight: ")
    
    return weights_df

def analyze_and_plot_performance(finaldf, regime_name, lookfwd, output_folder="Adarsh/output"):
    """
    Analyze performance and create visualizations for tactical vs static strategies.
    
    Parameters:
    - finaldf: DataFrame with tactical_next, static_next, Date, Regime, and Weight columns
    - regime_name: Name of the regime model (e.g., 'kmeans', 'gmm', 'hmm')
    - lookfwd: lookfwd period in months
    - output_folder: Folder to save outputs (default: 'Adarsh/output')
    """
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Create file prefix
    file_prefix = f"{regime_name}_lb{lookfwd}"
    
    # Convert period → datetime 
    finaldf = finaldf.copy()
    finaldf["Date"] = pd.to_datetime(finaldf["Date"].astype(str)) 
    
    # Cumulative 
    finaldf["tactical_cum"] = (1+finaldf["tactical_next"]).cumprod() 
    finaldf["static_cum"]   = (1+finaldf["static_next"]).cumprod() 
    
    # Metrics 
    def perf_stats(x): 
        rf=0 
        ann = (x.mean()-rf)*12 
        vol = x.std()*np.sqrt(12) 
        sharpe = ann/vol 
        mdd = ( ((1+x).cumprod().cummax() - (1+x).cumprod())/(1+x).cumprod().cummax() ).max() 
        return pd.Series({"AnnRet":ann,"AnnVol":vol,"Sharpe":sharpe,"MaxDD":mdd}) 

    tactical_stats = perf_stats(finaldf["tactical_next"].dropna())
    static_stats = perf_stats(finaldf["static_next"].dropna())

    # === Excess Return (Tactical − Static) ===
    tac = finaldf["tactical_next"].dropna()
    stat = finaldf["static_next"].dropna()

    excess_ret = (tac.mean() - stat.mean()) * 12

    # === Tracking Error ===
    active_returns = tac - stat
    tracking_error = active_returns.std() * np.sqrt(12)

    # === Information Ratio ===
    info_ratio = excess_ret / tracking_error if tracking_error != 0 else np.nan


    # === Create summary table ===
    excesssummary = pd.DataFrame({
        "Metric": ["Excess Return", "Tracking Error", "Information Ratio"],
        "Value": [
            f"{(excess_ret * 100).round(2)}%",       # percent
            f"{(tracking_error * 100).round(2)}%",   # percent
            info_ratio.round(2)                      # ratio
        ]
    })

    # === Save table as image ===
    fig, ax = plt.subplots(figsize=(6, 2))
    ax.axis('off')
    table = ax.table(
        cellText=excesssummary.values,
        colLabels=excesssummary.columns,
        cellLoc='center',
        loc='center'
    )

    table.scale(1, 2)
    plt.tight_layout()
    plot_path = os.path.join(output_folder, f"{file_prefix}_excessperf.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")
    plt.close()

    # Create figure with two subplots
    fig = plt.figure(figsize=(16, 7))
    gs = fig.add_gridspec(1, 2, width_ratios=[2.5, 1], wspace=0.3)

    # Plot 1: Cumulative Returns with Regime Shading
    ax1 = fig.add_subplot(gs[0])

    # Define regime colors (adjust based on your regime labels)
    regime_colors = {
        0: ('yellow', 'Regime 0'),
        1: ('lightskyblue', 'Regime 1'),
        2: ('lightgreen', 'Regime 2'),
        3: ('lightcoral', 'Regime 3'),
        4: ('plum', 'Regime 4'),
        5: ('purple', 'Regime 5')
    }

    # Add regime background shading
    if 'Regime' in finaldf.columns:
        # Reset index to work with integer positions
        finaldf_reset = finaldf.reset_index(drop=True)
        
        # Identify regime changes
        finaldf_reset['regime_change'] = finaldf_reset['Regime'] != finaldf_reset['Regime'].shift(1)
        regime_starts = finaldf_reset[finaldf_reset['regime_change']].index.tolist()
        
        if not regime_starts or regime_starts[0] != 0:
            regime_starts.insert(0, 0)  # Add first index if not present
        
        # Add last index + 1 to handle final regime
        regime_starts.append(len(finaldf_reset))
        
        # Track which regimes we've labeled
        labeled_regimes = set()
        
        # Create shaded regions for each regime
        for i in range(len(regime_starts) - 1):
            start_idx = regime_starts[i]
            end_idx = regime_starts[i + 1]  # Don't subtract 1 - go up to next regime start
            
            # Ensure valid indices
            if end_idx >= len(finaldf_reset):
                end_idx = len(finaldf_reset) - 1
            
            regime_val = finaldf_reset.loc[start_idx, 'Regime']
            start_date = finaldf_reset.loc[start_idx, 'Date']
            
            # Extend to the next regime's start date (or end of data)
            if i < len(regime_starts) - 2:
                # Use next regime's start date
                end_date = finaldf_reset.loc[regime_starts[i + 1], 'Date']
            else:
                # Last regime - extend to end of data
                end_date = finaldf_reset.loc[len(finaldf_reset) - 1, 'Date']
            
            if regime_val in regime_colors:
                color, label = regime_colors[regime_val]
                # Only add label for first occurrence of each regime
                label_to_use = label if regime_val not in labeled_regimes else None
                if label_to_use:
                    labeled_regimes.add(regime_val)
                
                ax1.axvspan(start_date, end_date, alpha=0.2, color=color, label=label_to_use)

    # Detect rebalancing events (when weights change)
    w_cols = [
        "Weight: Russell 1000 Value",
        "Weight: Russell 1000 Growth",
        "Weight: Russell 2000",
        "Weight: US Short-term Treasury",
        "Weight: US Long-term Treasury",
        "Weight: US IG Corporate Bond",
        "Weight: US HY Corporate Bond"
    ]

    if all(col in finaldf.columns for col in w_cols):
        # Create a column indicating if weights changed from previous month
        finaldf['weight_changed'] = ((finaldf[w_cols]).round(2) != (finaldf[w_cols].shift(1)).round(2)).any(axis=1)
        
        # Shift by 1 to show arrow at implementation date (one month after weight change)
        finaldf['show_rebalance_arrow'] = finaldf['weight_changed'].shift(1).fillna(False)
        
        # Get dates and cumulative values where arrows should be shown
        rebalance_mask = finaldf['show_rebalance_arrow'] == True
        rebalance_dates = finaldf.loc[rebalance_mask, 'Date'].tolist()
        rebalance_cums = finaldf.loc[rebalance_mask, 'tactical_cum'].tolist()

        # Plot cumulative returns
        ax1.plot(finaldf["Date"], finaldf["tactical_cum"], label="TACTICAL", 
                linewidth=2.5, color='#1f77b4', zorder=10)
        ax1.plot(finaldf["Date"], finaldf["static_cum"], label="STATIC", 
                linewidth=2.5, color='#ff7f0e', zorder=10)

    # Add rebalancing markers after plotting lines to get proper y-limits
    if all(col in finaldf.columns for col in w_cols):
        # Recalculate y-limits after plotting
        y_min, y_max = ax1.get_ylim()
        y_range = y_max - y_min
        
        # Add red arrows for rebalancing events
        rebalance_added = False
        for date, cum_val in zip(rebalance_dates, rebalance_cums):
            # Arrow points down from slightly above the line
            arrow_y = cum_val + y_range * 0.05
            
            ax1.annotate('', xy=(date, cum_val), xytext=(date, arrow_y),
                        arrowprops=dict(arrowstyle='->', color='red', lw=1.5, alpha=0.7),
                        zorder=15)
            
            # Add a small marker at arrow tip
            if not rebalance_added:
                ax1.plot([], [], 'rv', markersize=8, label='Rebalancing', alpha=0.7)
                rebalance_added = True

    # Styling
    ax1.legend(fontsize=11, loc='upper left', framealpha=0.9)
    ax1.set_title(f"Cumulative Returns – Tactical vs Static ({regime_name.upper()}, {lookfwd}M lookfwd)", 
                fontsize=13, fontweight='bold')
    ax1.set_xlabel("Date", fontsize=11)
    ax1.set_ylabel("Cumulative Return", fontsize=11)
    ax1.grid(True, alpha=0.3, zorder=0)

    # Plot 2: Summary Statistics Table
    ax2 = fig.add_subplot(gs[1])
    ax2.axis('tight')
    ax2.axis('off')

    # Prepare table data
    stats_data = [
        ['Metric', 'TACTICAL', 'STATIC'],
        ['Ann. Return', f'{tactical_stats["AnnRet"]:.2%}', f'{static_stats["AnnRet"]:.2%}'],
        ['Ann. Vol', f'{tactical_stats["AnnVol"]:.2%}', f'{static_stats["AnnVol"]:.2%}'],
        ['Sharpe', f'{tactical_stats["Sharpe"]:.3f}', f'{static_stats["Sharpe"]:.3f}'],
        ['Max DD', f'{tactical_stats["MaxDD"]:.2%}', f'{static_stats["MaxDD"]:.2%}']
    ]

    # Create table
    table = ax2.table(cellText=stats_data, cellLoc='center', loc='center',
                    colWidths=[0.35, 0.325, 0.325])

    # Style the table
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)

    # Color header row
    for i in range(3):
        cell = table[(0, i)]
        cell.set_facecolor('#4472C4')
        cell.set_text_props(weight='bold', color='white')

    # Alternate row colors
    for i in range(1, len(stats_data)):
        for j in range(3):
            cell = table[(i, j)]
            if i % 2 == 0:
                cell.set_facecolor('#E7E6E6')
            else:
                cell.set_facecolor('#F2F2F2')
            
            # Bold the metric names
            if j == 0:
                cell.set_text_props(weight='bold')

    ax2.set_title("Performance Summary", fontsize=13, fontweight='bold', pad=20)

    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(output_folder, f"{file_prefix}_performance.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved: {plot_path}")
    plt.close()

    # Save statistics to text file
    stats_path = os.path.join(output_folder, f"{file_prefix}_stats.txt")
    with open(stats_path, 'w') as f:
        f.write("="*50 + "\n")
        f.write(f"PERFORMANCE SUMMARY - {regime_name.upper()} (lookfwd: {lookfwd}M)\n")
        f.write("="*50 + "\n\n")
        
        f.write("TACTICAL:\n")
        f.write(tactical_stats.to_string() + "\n\n")
        
        f.write("STATIC:\n")
        f.write(static_stats.to_string() + "\n")
        f.write("="*50 + "\n")

        # Print regime statistics
        if 'Regime' in finaldf.columns:
            f.write("\n" + "="*50 + "\n")
            f.write("REGIME BREAKDOWN\n")
            f.write("="*50 + "\n")
            for regime in sorted(finaldf['Regime'].unique()):
                regime_data = finaldf[finaldf['Regime'] == regime]
                f.write(f"\nRegime {regime}: {len(regime_data)} months\n")
                f.write(f"  Date range: {regime_data['Date'].min().date()} to {regime_data['Date'].max().date()}\n")
                if 'tactical_next' in regime_data.columns:
                    regime_perf = perf_stats(regime_data['tactical_next'].dropna())
                    f.write(f"  Tactical Sharpe: {regime_perf['Sharpe']:.3f}\n")
            f.write("="*50 + "\n")

        # Print rebalancing statistics
        if all(col in finaldf.columns for col in w_cols):
            weight_changes = (finaldf[w_cols] != finaldf[w_cols].shift(1)).any(axis=1)
            num_rebalances = weight_changes.sum()
            f.write("\n" + "="*50 + "\n")
            f.write("REBALANCING ACTIVITY\n")
            f.write("="*50 + "\n")
            f.write(f"Total rebalancing events: {num_rebalances}\n")
            f.write(f"Average months between rebalances: {len(finaldf) / max(num_rebalances, 1):.1f}\n")
            f.write("="*50 + "\n")
    
    print(f"Statistics saved: {stats_path}")

    # Save data to CSV
    csv_path = os.path.join(output_folder, f"{file_prefix}_data.csv")
    finaldf.to_csv(csv_path, index=False)
    print(f"Data saved: {csv_path}")
    
    # Return stats for further analysis if needed
    return {
        'tactical_stats': tactical_stats,
        'static_stats': static_stats,
        'regime_name': regime_name,
        'lookfwd': lookfwd
    }

def plot_weights_over_time(finaldf, regime_name, lookfwd, output_folder="Adarsh/output"):
    """
    Plot portfolio weights over time as a stacked area chart.
    
    Parameters:
    - finaldf: DataFrame with Date, Regime, and Weight columns
    - regime_name: Name of the regime model (e.g., 'kmeans', 'gmm', 'hmm')
    - lookfwd: lookfwd period in months
    - output_folder: Folder to save outputs (default: 'Adarsh/output')
    """
    
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Create file prefix
    file_prefix = f"{regime_name}_lb{lookfwd}"
    
    # Convert period to datetime if needed
    finaldf = finaldf.copy()
    if not pd.api.types.is_datetime64_any_dtype(finaldf["Date"]):
        finaldf["Date"] = pd.to_datetime(finaldf["Date"].astype(str))
    
    # Define weight columns
    w_cols = [
        "Weight: Russell 1000 Value",
        "Weight: Russell 1000 Growth",
        "Weight: Russell 2000",
        "Weight: US Short-term Treasury",
        "Weight: US Long-term Treasury",
        "Weight: US IG Corporate Bond",
        "Weight: US HY Corporate Bond"
    ]
    
    # Extract weight data
    weights_df = finaldf[["Date"] + w_cols].copy()
    weights_df = weights_df.sort_values("Date")
    
    # Create the plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # Define colors for each asset
    colors = {
        "Weight: Russell 1000 Value": '#5B9BD5',      # Blue
        "Weight: Russell 1000 Growth": '#F4B183',     # Orange
        "Weight: Russell 2000": '#C55A5A',            # Red
        "Weight: US Short-term Treasury": '#A5A5A5',  # Gray
        "Weight: US Long-term Treasury": '#C27BA0',   # Pink/Purple
        "Weight: US IG Corporate Bond": '#70AD47',    # Green
        "Weight: US HY Corporate Bond": '#4BACC6'     # Cyan
    }
    
    # Create stacked area chart
    ax.stackplot(
        weights_df["Date"],
        *[weights_df[col] for col in w_cols],
        labels=[col.replace("Weight: ", "") for col in w_cols],
        colors=[colors[col] for col in w_cols],
        alpha=0.85
    )
    
    # Styling
    ax.set_ylim(0, 1)
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Weight", fontsize=12)
    ax.set_title(f"Portfolio Weight Changes Over Time (Monthly Rebalancing)\n{regime_name.upper()}, {lookfwd}M lookfwd", 
                 fontsize=14, fontweight='bold')
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    # Save figure
    plot_path = os.path.join(output_folder, f"{file_prefix}_weights_over_time.png")
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Weights plot saved: {plot_path}")
    plt.close()


## Input data
if __name__=="__main__":
    russell2000=pd.read_csv("Russel2000data.csv")
    russell2000['Russell 2000'] = (russell2000['RTY Index'] - russell2000['RTY Index'].shift(-1)) / russell2000['RTY Index'].shift(-1)
    russell2000["Date"]=pd.to_datetime(russell2000["Date"]).dt.to_period("M")
    monthly_ret= pd.read_csv("df_1M_ret.csv")
    monthly_ret["Date"]=pd.to_datetime(monthly_ret["Date"]).dt.to_period("M")
    monthly_ret= monthly_ret.merge(russell2000, on="Date", how="left")

    regimeslist=["kmeans", "gmm", "hmm"]
    lookfwds=[6,12]

    # BASE (strategic neutral) weights
    base = pd.Series({
        "Russell 1000 Value": 0.25,
        "Russell 1000 Growth": 0.25,
        "Russell 2000": 0.10,
        "US Short-term Treasury": 0.10,
        "US Long-term Treasury": 0.10,
        "US IG Corporate Bond": 0.15,
        "US HY Corporate Bond": 0.05
    })

    # Store all results
    all_results = []

    # Define column names once
    w_cols = [
        "Weight: Russell 1000 Value",
        "Weight: Russell 1000 Growth",
        "Weight: Russell 2000",
        "Weight: US Short-term Treasury",
        "Weight: US Long-term Treasury",
        "Weight: US IG Corporate Bond",
        "Weight: US HY Corporate Bond"
    ]
    r_cols = [
        "Russell 1000 Value",
        "Russell 1000 Growth",
        "Russell 2000",
        "US Short-term Treasury",
        "US Long-term Treasury",
        "US IG Corporate Bond",
        "US HY Corporate Bond"
    ]

    for regime in regimeslist:
        startdf = load_regime_data(regimename=regime)
        
        # Get all dates post-2020 for out-of-sample testing
        dfpost2020 = startdf[startdf["Date"] >= "2020-01-01"].copy()
        post2020_dates = dfpost2020["Date"].unique()
        
        for lookfwd in lookfwds:
            print(f"Processing {regime} with {lookfwd}-month lookforward...")
            
            # For each month in test period, calculate weights using expanding window
            monthly_weights_list = []
            
            for current_date in post2020_dates:
                # Use data up to (but not including) current month for training
                cutoff = current_date
                train_data = startdf[startdf["Date"] < cutoff].copy()
                
                if len(train_data) < lookfwd:
                    # Not enough history, skip
                    continue
                
                # Calculate regime performance on training data
                result_fwd = get_regime_perf(train_data, forwardlook=lookfwd)
                
                if len(result_fwd) == 0:
                    continue
                
                # Generate weights for each regime
                weights_df = generate_dynamic_weights_v2(result_fwd, base)
                
                # Get current month's regime
                current_regime = dfpost2020[dfpost2020["Date"] == current_date]["Regime"].values
                
                if len(current_regime) == 0:
                    continue
                    
                current_regime = current_regime[0]
                
                # Get weights for current regime
                if current_regime in weights_df.index:
                    current_weights = weights_df.loc[current_regime].copy()
                    current_weights["Date"] = current_date
                    current_weights["Regime"] = current_regime
                    current_weights["lookfwd"] = lookfwd
                    current_weights["regime_model"] = regime
                    monthly_weights_list.append(current_weights)
            
            if len(monthly_weights_list) > 0:
                # Combine all monthly weights
                monthly_weights_df = pd.DataFrame(monthly_weights_list)
                
                # Merge with actual returns
                finaldf = dfpost2020.merge(monthly_weights_df, on=["Date", "Regime"], how="inner")
                
                # Calculate portfolio returns
                # Tactical: use lagged weights (weights determined in previous month)
                finaldf["tactical_next"] = (
                    finaldf[r_cols].values * finaldf[w_cols].shift(1).values
                ).sum(axis=1)

                # Static benchmark: use base weights
                static_w = base.values  # Use the same base weights for consistency
                finaldf["static_next"] = (
                    finaldf[r_cols].values * static_w
                ).sum(axis=1)
                
                # Store result
                result_dict = {
                    'regime_model': regime,
                    'lookfwd': lookfwd,
                    'data': finaldf
                }
                all_results.append(result_dict)
                
                print(f"  Completed: {len(finaldf)} months of data")
                
                # Analyze and plot performance
                stats = analyze_and_plot_performance(
                    finaldf=finaldf.dropna(subset=["tactical_next"]), 
                    regime_name=regime,
                    lookfwd=lookfwd
                )

                # After the analyze_and_plot_performance call, add:
                plot_weights_over_time(
                    finaldf=finaldf.dropna(subset=["tactical_next"]), 
                    regime_name=regime,
                    lookfwd=lookfwd
                )