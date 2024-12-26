import requests
from bs4 import BeautifulSoup
import pandas as pd

# 目标URL
url = "https://directory.seas.upenn.edu/"

# 发送请求获取网页内容
response = requests.get(url)

# 检查请求是否成功
if response.status_code == 200:
    # 使用BeautifulSoup解析HTML文档
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # 查找所有包含StaffListMeta类的div元素
    staff_list_meta = soup.find_all('div', class_='StaffListMeta')

    # 创建列表以保存所有工作人员的数据
    staff_data = []

    for staff in staff_list_meta:
        # 提取姓名和对应的个人主页链接
        name_link = staff.find('div', class_='StaffListName').find('a', href=True)
        name = name_link.get_text(strip=True)
        personal_page = name_link['href']
        
        # 提取职位和部门
        titles_div = staff.find('div', class_='StaffListTitles')
        titles = [title.get_text(strip=True) for title in titles_div.find_all('div')] if titles_div else []
        
        # 提取特别职务
        special_title = staff.find('div', class_='StaffListSpecialTitle').get_text(strip=True) if staff.find('div', class_='StaffListSpecialTitle') else None
        
        # 提取电子邮件链接
        email_link = staff.find('a', href=True)['href'] if staff.find('a', href=True) and 'mailto:' in staff.find('a', href=True)['href'] else None
        
        # 将信息添加到列表中
        staff_data.append({
            '姓名': name,
            '个人主页': personal_page,
            '职位和部门': ' | '.join(titles),
            '特别职务': special_title or '无',
            '电子邮件': email_link
        })
    
    # 创建数据框
    df = pd.DataFrame(staff_data)

    # 将数据框保存为Excel文件
    excel_file_path = 'staff_list_with_personal_pages.xlsx'
    df.to_excel(excel_file_path, index=False)
    
    print(f"数据已成功保存到 {excel_file_path}")
else:
    print("无法获取网页内容，状态码:", response.status_code)