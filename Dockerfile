FROM python:3.7-slim

# Update source
RUN sed -i s@/deb.debian.org/@/mirrors.aliyun.com/@g /etc/apt/sources.list \
&& apt-get clean \
&& apt-get update 

# Install Java
RUN apt-get update && apt-get install -y default-jdk
RUN apt-get update && apt-get install -y wget
RUN apt-get update && apt-get install -y vim
RUN apt-get update && apt-get install -y procps

# Install Tomcat
# Download and extract the Tomcat binary
RUN wget https://mirrors.cnnic.cn/apache/tomcat/tomcat-9/v9.0.70/bin/apache-tomcat-9.0.70.tar.gz
RUN tar -xzvf apache-tomcat-9.0.70.tar.gz
# Rename the extracted folder to "tomcat"
RUN mv apache-tomcat-9.0.70 tomcat
CMD ["cd","/tomcat/bin"]
CMD ["./startup.sh"]

# Install Mysql
# Install MySQL server
RUN apt-get update && apt-get install -y mariadb-server mariadb-client
# Run MySQL server
#CMD ["mysqld"]
# Create testdb database
#RUN service mysql start && \
#    mysql -u root -e "CREATE DATABASE testdb;

# Install Nginx
RUN apt-get update
RUN apt-get update && apt-get install -y \
    nginx 
COPY nginx.conf /etc/nginx/nginx.conf
CMD ["nginx", "-g", "daemon off;"]

# Install PyTorch
RUN pip install torch -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
RUN pip install pfl -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
RUN pip install requests -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
RUN pip install pandas -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
RUN pip install flask -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
RUN pip install matplotlib -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
RUN pip install torchvision  -i http://pypi.douban.com/simple --trusted-host pypi.douban.com
# Expose the default Tomcat port
EXPOSE 8080
#

# Start Tomcat
COPY show.html /tomcat/webapps/ROOT
RUN cd /tomcat/bin
# Start APP
COPY /app /usr/app
WORKDIR /usr/app
CMD ["python","fl_model.py"]
CMD ["python","fl_server.py"]
CMD ["python","fl_client1.py"]
CMD ["python","fl_client2.py"]
CMD ["python","fl_client3.py"]
