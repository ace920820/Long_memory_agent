import pkg_resources
import sys

def generate_requirements():
    # 项目实际需要的包
    required_packages = {
        # Web Framework
        'flask',
        
        # AI/ML Libraries
        'faiss-cpu',
        'sentence-transformers',
        'numpy',
        'torch',
        'transformers',
        
        # Configuration
        'pyyaml',
        
        # Logging
        'python-json-logger',
        
        # Utilities
        'python-dateutil',
        'requests',
        'tqdm',
        
        # Development Tools
        'pytest',
        'black',
        'flake8'
    }
    
    # 获取当前环境中已安装的包
    installed_packages = {dist for dist in pkg_resources.working_set}
    
    # 创建或覆盖 requirements.txt，按类别组织
    with open('requirements.txt', 'w', encoding='utf-8') as f:
        # Web Framework
        f.write("# Web Framework\n")
        for package in installed_packages:
            if package.key.lower() == 'flask':
                f.write(f'{package.key}=={package.version}\n')
        f.write("\n")
        
        # AI/ML Libraries
        f.write("# AI/ML Libraries\n")
        ai_libs = {'faiss-cpu', 'sentence-transformers', 'numpy', 'torch', 'transformers'}
        for package in installed_packages:
            if package.key.lower() in ai_libs:
                f.write(f'{package.key}=={package.version}\n')
        f.write("\n")
        
        # Configuration
        f.write("# Configuration\n")
        for package in installed_packages:
            if package.key.lower() == 'pyyaml':
                f.write(f'{package.key}=={package.version}\n')
        f.write("\n")
        
        # Logging
        f.write("# Logging\n")
        for package in installed_packages:
            if package.key.lower() == 'python-json-logger':
                f.write(f'{package.key}=={package.version}\n')
        f.write("\n")
        
        # Utilities
        f.write("# Utilities\n")
        utils = {'python-dateutil', 'requests', 'tqdm'}
        for package in installed_packages:
            if package.key.lower() in utils:
                f.write(f'{package.key}=={package.version}\n')
        f.write("\n")
        
        # Development Tools
        f.write("# Development Tools\n")
        dev_tools = {'pytest', 'black', 'flake8'}
        for package in installed_packages:
            if package.key.lower() in dev_tools:
                f.write(f'{package.key}=={package.version}\n')

    print("requirements.txt 已生成完成！")

if __name__ == "__main__":
    generate_requirements() 