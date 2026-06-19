import base64
from io import BytesIO

import numpy as np
import pandas as pd


try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import seaborn as sns
    PLOT_AVAILABLE = True
except Exception:
    PLOT_AVAILABLE = False


ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def find_numeric_columns(df):
    return df.select_dtypes(include=[np.number]).columns.tolist()

def find_categorical_columns(df):
    return df.select_dtypes(include=['object']).columns.tolist()


def generate_eda_report(df):
    try:
        eda_data = {}
        eda_data['shape'] = {'rows': len(df), 'columns': len(df.columns)}
        eda_data['memory'] = f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
        missing = df.isnull().sum()
        eda_data['missing_values'] = {
            col: {
                'count': int(missing[col]),
                'percentage': f"{(missing[col] / len(df) * 100):.2f}%"
            }
            for col in df.columns if missing[col] > 0
        }
        eda_data['data_types'] = df.dtypes.astype(str).to_dict()
        eda_data['duplicates'] = {
            'total_duplicates': int(df.duplicated().sum()),
            'duplicate_percentage': f"{(df.duplicated().sum() / len(df) * 100):.2f}%"
        }
        numeric_cols = find_numeric_columns(df)
        eda_data['numeric_stats'] = {}
        for col in numeric_cols:
            col_data = df[col].dropna()
            eda_data['numeric_stats'][col] = {
                'count': len(col_data),
                'mean': f"{col_data.mean():.2f}",
                'median': f"{col_data.median():.2f}",
                'std': f"{col_data.std():.2f}",
                'min': f"{col_data.min():.2f}",
                'max': f"{col_data.max():.2f}",
                'q1': f"{col_data.quantile(0.25):.2f}",
                'q3': f"{col_data.quantile(0.75):.2f}",
                'skewness': f"{col_data.skew():.2f}",
                'kurtosis': f"{col_data.kurtosis():.2f}"
            }
        cat_cols = find_categorical_columns(df)
        eda_data['categorical_stats'] = {}
        for col in cat_cols:
            eda_data['categorical_stats'][col] = {
                'unique': int(df[col].nunique()),
                'top': str(df[col].mode()[0]) if len(df[col].mode()) > 0 else "N/A",
                'frequency': int(df[col].value_counts().iloc[0]) if len(df[col].value_counts()) > 0 else 0
            }
        return eda_data
    except Exception as e:
        return {'error': str(e)}


def create_correlation_plot(df):
    try:
        if not PLOT_AVAILABLE:
            return None
        numeric_cols = find_numeric_columns(df)
        if len(numeric_cols) < 2:
            return None
        plt.figure(figsize=(8, 6))
        corr = df[numeric_cols].corr()
        sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0, square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{image_base64}"
    except Exception:
        return None


def create_distribution_plots(df):
    try:
        if not PLOT_AVAILABLE:
            return None
        numeric_cols = find_numeric_columns(df)
        if len(numeric_cols) == 0:
            return None
        n_cols = min(3, len(numeric_cols))
        n_rows = (len(numeric_cols) + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
        if n_rows == 1 and n_cols == 1:
            axes = np.array([axes])
        elif n_rows == 1 or n_cols == 1:
            axes = axes.flatten()
        else:
            axes = axes.flatten()
        for idx, col in enumerate(numeric_cols):
            ax = axes[idx]
            df[col].dropna().hist(bins=30, ax=ax, color='steelblue', edgecolor='black')
            ax.set_title(f'Distribution of {col}', fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Frequency')
            ax.grid(alpha=0.3)
        for idx in range(len(numeric_cols), len(axes)):
            axes[idx].axis('off')
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{image_base64}"
    except Exception:
        return None


def create_missing_plot(df):
    try:
        if not PLOT_AVAILABLE:
            return None
        missing = (df.isnull().sum() / len(df) * 100).sort_values(ascending=False)
        if missing.sum() == 0:
            return None
        missing = missing[missing > 0]
        plt.figure(figsize=(10, 6))
        missing.plot(kind='barh', color='coral', edgecolor='black')
        plt.title('Missing Values (%)', fontweight='bold', fontsize=14)
        plt.xlabel('Percentage Missing (%)')
        plt.grid(alpha=0.3, axis='x')
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{image_base64}"
    except Exception:
        return None


def create_boxplots(df):
    try:
        if not PLOT_AVAILABLE:
            return None
        numeric_cols = find_numeric_columns(df)
        if len(numeric_cols) == 0:
            return None
        fig, axes = plt.subplots(1, min(3, len(numeric_cols)), figsize=(15, 5))
        if len(numeric_cols) == 1:
            axes = [axes]
        for idx, col in enumerate(numeric_cols[:3]):
            df[col].dropna().plot(kind='box', ax=axes[idx], color='lightblue')
            axes[idx].set_title(f'Boxplot of {col}', fontweight='bold')
            axes[idx].set_ylabel(col)
            axes[idx].grid(alpha=0.3)
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{image_base64}"
    except Exception:
        return None


def create_histogram(df, column):
    try:
        if not PLOT_AVAILABLE:
            return None
        if column not in df.columns:
            return None
        if df[column].dtype not in ['int64', 'float64', 'int32', 'float32', 'int', 'float']:
            try:
                df[column] = pd.to_numeric(df[column])
            except Exception:
                return None
        plt.figure(figsize=(10, 6))
        data = df[column].dropna()
        if len(data) == 0:
            return None
        plt.hist(data, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        plt.title(f'Histogram of {column}', fontweight='bold', fontsize=14)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3, axis='y')
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{image_base64}"
    except Exception:
        return None


def create_boxplot(df, column):
    try:
        if not PLOT_AVAILABLE:
            return None
        if column not in df.columns:
            return None
        if df[column].dtype not in ['int64', 'float64', 'int32', 'float32', 'int', 'float']:
            try:
                df[column] = pd.to_numeric(df[column])
            except Exception:
                return None
        plt.figure(figsize=(10, 6))
        data = df[column].dropna()
        if len(data) == 0:
            return None
        plt.boxplot(data, vert=True)
        plt.title(f'Boxplot of {column}', fontweight='bold', fontsize=14)
        plt.ylabel(column)
        plt.grid(alpha=0.3, axis='y')
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{image_base64}"
    except Exception:
        return None


def create_scatter_plot(df, column=None):
    try:
        if not PLOT_AVAILABLE:
            return None
        numeric_cols = find_numeric_columns(df)
        if len(numeric_cols) < 2:
            return None
        x_col = numeric_cols[0]
        y_col = numeric_cols[1] if len(numeric_cols) > 1 else numeric_cols[0]
        if column and column in numeric_cols:
            x_col = column
            remaining = [c for c in numeric_cols if c != column]
            if remaining:
                y_col = remaining[0]
        x_data = df[x_col].dropna()
        y_data = df[y_col].dropna()
        if len(x_data) == 0 or len(y_data) == 0:
            return None, None, None
        plt.figure(figsize=(10, 6))
        plt.scatter(x_data, y_data, alpha=0.6, color='steelblue', s=50)
        plt.title(f'Scatter Plot: {x_col} vs {y_col}', fontweight='bold', fontsize=14)
        plt.xlabel(x_col)
        plt.ylabel(y_col)
        plt.grid(alpha=0.3)
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{image_base64}", x_col, y_col
    except Exception:
        return None, None, None


def create_bar_chart_with_cols(df, column1, column2=None):
    try:
        if not PLOT_AVAILABLE:
            return None
        if column1 not in df.columns:
            return None
        if column2 and column2 in df.columns:
            grouped = df.groupby([column1, column2]).size().unstack(fill_value=0)
            if len(grouped) == 0:
                return None
            plt.figure(figsize=(12, 6))
            grouped.plot(kind='bar', ax=plt.gca(), edgecolor='black')
            plt.title(f'{column1} vs {column2}', fontweight='bold', fontsize=14)
            plt.xlabel(column1)
            plt.ylabel('Count')
            plt.legend(title=column2, bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.xticks(rotation=45, ha='right')
            plt.grid(alpha=0.3, axis='y')
            plt.tight_layout()
        else:
            value_counts = df[column1].value_counts().head(10)
            if len(value_counts) == 0:
                return None
            plt.figure(figsize=(10, 6))
            value_counts.plot(kind='bar', color='steelblue', edgecolor='black')
            plt.title(f'Top 10 - {column1}', fontweight='bold', fontsize=14)
            plt.xlabel(column1)
            plt.ylabel('Count')
            plt.xticks(rotation=45, ha='right')
            plt.grid(alpha=0.3, axis='y')
            plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{image_base64}"
    except Exception:
        return None


def create_count_plot(df, column):
    try:
        if not PLOT_AVAILABLE:
            return None
        if column not in df.columns:
            return None
        value_counts = df[column].value_counts()
        if len(value_counts) == 0:
            return None
        plt.figure(figsize=(10, 6))
        colors = sns.color_palette("husl", len(value_counts))
        plt.barh(value_counts.index[::-1], value_counts.values[::-1], color=colors)
        plt.title(f'Count Distribution - {column}', fontweight='bold', fontsize=14)
        plt.xlabel('Count')
        plt.grid(alpha=0.3, axis='x')
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{image_base64}"
    except Exception:
        return None


def create_pie_chart(df, column):
    try:
        if not PLOT_AVAILABLE:
            return None
        if column not in df.columns:
            return None
        value_counts = df[column].value_counts()
        if len(value_counts) == 0:
            return None
        plt.figure(figsize=(10, 8))
        colors = sns.color_palette("husl", len(value_counts))
        if len(value_counts) > 8:
            top_counts = value_counts.head(8)
            other_count = value_counts[8:].sum()
            top_counts = pd.concat([top_counts, pd.Series({'Others': other_count})])
        else:
            top_counts = value_counts
        wedges, texts, autotexts = plt.pie(
            top_counts.values,
            labels=top_counts.index,
            autopct='%1.1f%%',
            colors=colors[:len(top_counts)],
            startangle=90,
            textprops={'fontsize': 9}
        )
        for autotext in autotexts:
            autotext.set_color('white')
            autotext.set_fontweight('bold')
            autotext.set_fontsize(9)
        plt.title(f'Distribution of {column}', fontweight='bold', fontsize=14, pad=20)
        plt.tight_layout()
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{image_base64}"
    except Exception:
        return None


def create_distribution_plot(df, column):
    try:
        if not PLOT_AVAILABLE:
            return None
        if column not in df.columns:
            return None
        if df[column].dtype not in ['int64', 'float64', 'int32', 'float32', 'int', 'float']:
            try:
                df[column] = pd.to_numeric(df[column])
            except Exception:
                return None
        data = df[column].dropna()
        if len(data) == 0:
            return None
        plt.figure(figsize=(10, 6))
        sns.histplot(data=data, kde=True, bins=30, color='steelblue', edgecolor='black', alpha=0.7)
        plt.title(f'Distribution of {column}', fontweight='bold', fontsize=14)
        plt.xlabel(column)
        plt.ylabel('Frequency')
        plt.grid(alpha=0.3, axis='y')
        buffer = BytesIO()
        plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
        buffer.seek(0)
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        plt.close()
        return f"data:image/png;base64,{image_base64}"
    except Exception:
        return None
