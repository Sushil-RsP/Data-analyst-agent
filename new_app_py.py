from flask import Flask, render_template, request, jsonify
import os
from werkzeug.utils import secure_filename
import pandas as pd

from new_analysis_tools import (
    allowed_file,
    find_numeric_columns,
    find_categorical_columns,
    generate_eda_report,
    create_correlation_plot,
    create_distribution_plots,
    create_missing_plot,
    create_boxplots,
    create_histogram,
    create_boxplot,
    create_scatter_plot,
    create_bar_chart_with_cols,
    create_count_plot,
    create_pie_chart,
    create_distribution_plot,
)


from gemini_pandas_model import (
    generate_pandas_code,
    execute_code,
    format_result
)

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

df = None


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_file():
    global df
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Only CSV and Excel files allowed'}), 400

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(filepath)
            elif filename.endswith('.xlsx') or filename.endswith('.xls'):
                df = pd.read_excel(filepath)
            else:
                raise ValueError("Only CSV and XLSX files supported")   
        except Exception as file_error:
            os.remove(filepath)
            return jsonify({'error': f'Error reading file: {str(file_error)}'}), 400

        df = df.copy().drop_duplicates().reset_index(drop=True)
        for col in df.columns:
            try:
                if pd.api.types.is_numeric_dtype(df[col]):
                    median_val = df[col].median()
                    if pd.notna(median_val):
                        df[col] = df[col].fillna(median_val)
                else:
                    mode_vals = df[col].mode()
                    if len(mode_vals) > 0:
                        df[col] = df[col].fillna(mode_vals[0])
                    else:
                        df[col] = df[col].fillna('Unknown')
            except Exception:
                continue

        summary = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns.tolist(),
            'column_types': df.dtypes.astype(str).to_dict(),
            'sample': df.head(5).to_html(classes="table table-sm", index=False)
        }

        try:
            os.remove(filepath)
        except Exception:
            pass

        return jsonify({'success': True, 'summary': summary})
    except Exception as e:
        return jsonify({'error': f'Upload failed: {str(e)}'}), 500


@app.route('/api/query', methods=['POST'])
def query():
    global df
    if df is None:
        return jsonify({'success': True, 'result': 'Please upload a file first.', 'intent': 'system'})

    data = request.get_json(silent=True) or {}
    question = data.get('question', '').strip()
    if not question:
        return jsonify({'success': True, 'result': 'Please enter a question.', 'intent': 'system'})

    try:
        code = generate_pandas_code(question, df)

        #print("\nGenerated Pandas Code:")
        #print(code)

        result = execute_code(code, df)
        
        result = format_result(result)

        #print("\nAnswer:")
        #print(format_result(result))

        return jsonify({'success': True, 'result': result})
    except Exception as e:
        return jsonify({'success': True, 'result': f"Something went wrong while analyzing that question: {str(e)}"})


@app.route('/api/dataframe-info')
def dataframe_info():
    global df
    if df is None:
        return jsonify({'error': 'No data loaded'}), 400
    info = {
        'shape': f"{len(df)} rows × {len(df.columns)} columns",
        'columns': df.columns.tolist(),
        'dtypes': df.dtypes.astype(str).to_dict(),
        'preview': df.head(10).to_html(classes="table table-sm", index=False)
    }
    return jsonify(info)


@app.route('/api/eda')
def get_eda():
    global df
    if df is None:
        return jsonify({'error': 'No data loaded'}), 400
    return jsonify(generate_eda_report(df))


@app.route('/api/eda-plots')
def get_eda_plots():
    global df
    if df is None:
        return jsonify({'error': 'No data loaded'}), 400

    plots = {
        'correlation': create_correlation_plot(df),
        'distribution': create_distribution_plots(df),
        'missing': create_missing_plot(df),
        'boxplot': create_boxplots(df)
    }
    plots = {k: v for k, v in plots.items() if v is not None}
    return jsonify(plots)


@app.route('/api/charts', methods=['POST'])
def generate_chart():
    global df
    if df is None:
        return jsonify({'error': 'No data loaded'}), 400

    data = request.get_json() or {}
    chart_type = data.get('chart_type', '').lower()
    column = data.get('column', '')
    column2 = data.get('column2', '')

    if not chart_type:
        return jsonify({'error': 'Chart type is required'}), 400

    try:
        result_column = column
        result_column2 = None
        chart = None

        if chart_type == 'histogram':
            if not column:
                return jsonify({'error': 'Column is required for histogram'}), 400
            chart = create_histogram(df, column)
        elif chart_type == 'boxplot':
            if not column:
                return jsonify({'error': 'Column is required for boxplot'}), 400
            chart = create_boxplot(df, column)
        elif chart_type == 'scatter plot':
            if not column or not column2:
                return jsonify({'error': 'Two columns are required for scatter plot'}), 400
            chart, x_col, y_col = create_scatter_plot(df, column)
            result_column = column
            result_column2 = column2
        elif chart_type == 'bar chart':
            if not column:
                return jsonify({'error': 'Column is required for bar chart'}), 400
            chart = create_bar_chart_with_cols(df, column, column2)
            result_column = column
            result_column2 = column2 if column2 else None
        elif chart_type == 'count plot':
            if not column:
                return jsonify({'error': 'Column is required for count plot'}), 400
            chart = create_count_plot(df, column)
        elif chart_type == 'pie chart':
            if not column:
                return jsonify({'error': 'Column is required for pie chart'}), 400
            chart = create_pie_chart(df, column)
        elif chart_type == 'distribution':
            if not column:
                return jsonify({'error': 'Column is required for distribution'}), 400
            chart = create_distribution_plot(df, column)
        elif chart_type == 'correlation heatmap':
            chart = create_correlation_plot(df)
            result_column = 'All Numeric Columns'
        else:
            return jsonify({'error': f'Unknown chart type: {chart_type}'}), 400

        if chart is None:
            return jsonify({'error': f'Could not generate {chart_type}. Ensure column exists and has appropriate data type.'}), 400

        return jsonify({
            'success': True,
            'chart': chart,
            'chart_type': chart_type,
            'column': result_column,
            'column2': result_column2
        })
    except Exception as e:
        return jsonify({'error': f'Server error: {str(e)}'}), 500


@app.route('/api/chart-columns')
def get_chart_columns():
    global df
    if df is None:
        return jsonify({'error': 'No data loaded'}), 400

    return jsonify({
        'numeric': find_numeric_columns(df),
        'categorical': find_categorical_columns(df),
        'all': df.columns.tolist()
    })


@app.route('/api/columns-for-chart/<chart_type>')
def get_columns_for_chart(chart_type):
    global df
    if df is None:
        return jsonify({'error': 'No data loaded'}), 400

    numeric_cols = find_numeric_columns(df)
    categorical_cols = find_categorical_columns(df)
    chart_type = chart_type.lower()

    column_mapping = {
        'histogram': numeric_cols,
        'boxplot': numeric_cols,
        'distribution': numeric_cols,
        'scatter plot': numeric_cols,
        'bar chart': categorical_cols + numeric_cols,
        'count plot': categorical_cols,
        'pie chart': categorical_cols,
        'correlation heatmap': []
    }

    columns = column_mapping.get(chart_type, [])
    min_required = 2 if chart_type == 'scatter plot' else 1

    return jsonify({
        'columns': columns,
        'min_required': min_required,
        'column_type': 'numeric' if chart_type in ['histogram', 'boxplot', 'distribution', 'scatter plot'] else 'categorical' if chart_type in ['count plot', 'pie chart'] else 'mixed'
    })


if __name__ == '__main__':
    app.run(debug=True, port=5000)
