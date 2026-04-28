import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# API Key
# LJzlMp4P.PTM265yevjzUEpi7FDyaqK9tBUWqH7UJ

def load_and_process_data(patent_file, cpc_file):
    """Load and process patent and CPC classification data"""
    # Read the data files
    patents_df = pd.read_csv(patent_file,
                             usecols=['patent_id', 'patent_date', 'country_code'])
    cpc_df = pd.read_csv(cpc_file,
                         usecols=['patent_id', 'cpc_group'])

    # Convert patent_date to datetime
    patents_df['patent_date'] = pd.to_datetime(patents_df['patent_date'])

    # Filter for Y02 patents
    y02_patents = cpc_df[cpc_df['cpc_group'].str.startswith('Y02', na=False)]

    # Merge patent data with CPC classifications
    merged_df = pd.merge(y02_patents, patents_df, on='patent_id')

    # Extract Y02 subcategory
    merged_df['y02_category'] = merged_df['cpc_group'].str[:4]

    return merged_df


def analyze_by_country_and_time(df, time_period='Y'):
    """Analyze patent trends by country and time period (Y=yearly, M=monthly)"""
    # Define major countries to analyze
    major_countries = ['US', 'CN', 'JP', 'KR', 'DE', 'FR', 'GB']

    # Filter for major countries
    df_major = df[df['country_code'].isin(major_countries)]

    # Group by time period and country
    df_major['period'] = df_major['patent_date'].dt.to_period(time_period)
    country_time_counts = df_major.groupby(['period', 'country_code']).size().unstack(fill_value=0)

    return country_time_counts


def analyze_technology_distribution(df):
    """Analyze distribution of Y02 subcategories"""
    category_map = {
        'Y02A': '适应气候变化的技术', # CCAT
        'Y02B': '建筑节能技术',  # CCMT
        'Y02C': '温室气体捕获与储存',
        'Y02D': 'ICT节能技术', # CCMT
        'Y02E': '能源生产减排技术',
        'Y02P': '生产工艺减排技术', # CCMT
        'Y02T': '交通运输减排技术', # CCMT
        'Y02W': '废物处理与循环利用' # CCMT
    }

    df['category_name'] = df['y02_category'].map(category_map)
    tech_distribution = df.groupby(['country_code', 'category_name']).size().unstack(fill_value=0)

    return tech_distribution


def plot_country_trends(country_time_counts, title):
    """Create line plot for country trends"""
    plt.figure(figsize=(12, 6))
    for country in country_time_counts.columns:
        plt.plot(country_time_counts.index.astype(str),
                 country_time_counts[country],
                 label=country,
                 marker='o')

    plt.title(title)
    plt.xlabel('时间')
    plt.ylabel('专利数量')
    plt.legend(title='国家')
    plt.xticks(rotation=45)
    plt.grid(True)
    return plt


def plot_technology_heatmap(tech_distribution):
    """Create heatmap for technology distribution by country"""
    plt.figure(figsize=(12, 8))
    sns.heatmap(tech_distribution,
                annot=True,
                fmt='g',
                cmap='YlOrRd')
    plt.title('各国绿色专利技术分布')
    plt.xlabel('技术类别')
    plt.ylabel('国家')
    return plt


# Main execution
def main(patent_file, cpc_file):
    print("正在加载和处理数据...")
    merged_data = load_and_process_data(patent_file, cpc_file)

    # Yearly analysis
    yearly_trends = analyze_by_country_and_time(merged_data, 'Y')
    print("\n年度专利统计:")
    print(yearly_trends.tail())

    # Monthly analysis for the recent period
    recent_data = merged_data[merged_data['patent_date'] >= '2023-01-01']
    monthly_trends = analyze_by_country_and_time(recent_data, 'M')
    print("\n最近月度专利统计:")
    print(monthly_trends.tail())

    # Technology distribution analysis
    tech_dist = analyze_technology_distribution(merged_data)
    print("\n技术分布统计:")
    print(tech_dist)

    # Create visualizations
    plot_country_trends(yearly_trends, '主要国家绿色专利年度趋势')
    plt.savefig('yearly_trends.png')

    plot_country_trends(monthly_trends, '主要国家绿色专利月度趋势')
    plt.savefig('monthly_trends.png')

    plot_technology_heatmap(tech_dist)
    plt.savefig('tech_distribution.png')


if __name__ == "__main__":
    # Update these paths to match your downloaded file locations
    patent_file = "path_to_your_patent_file.csv"
    cpc_file = "path_to_your_cpc_file.csv"
    main(patent_file, cpc_file)