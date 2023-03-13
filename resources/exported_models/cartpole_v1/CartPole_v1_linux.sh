#!/bin/sh
# 
# Run AnyLogic Experiment
# 
DIR_BACKUP_XJAL=$(pwd)
SCRIPT_DIR_XJAL=$(dirname "$0")
cd "$SCRIPT_DIR_XJAL"
chmod +x chromium/chromium-linux64/chrome

java -cp model.jar:lib/MarkupDescriptors.jar:lib/model0.jar:resources:lib/unirest-java-3.14.1.jar:lib/slf4j-api-1.7.25.jar:lib/py4j0.10.7.jar:lib/logback-core-1.2.3.jar:lib/logback-classic-1.2.3.jar:lib/junit-4.12.jar:lib/httpmime-4.5.13.jar:lib/httpcore-nio-4.4.13.jar:lib/httpcore-4.4.13.jar:lib/httpclient-4.5.13.jar:lib/httpasyncclient-4.1.5.jar:lib/hamcrest-core-1.3.jar:lib/gson-2.9.0.jar:lib/commons-logging-1.2.jar:lib/commons-codec-1.15.jar:lib/commons-cli-1.3.1.jar:lib/unirest-java-3.14.1.jar:lib/httpmime-4.5.13.jar:lib/httpcore-nio-4.4.13.jar:lib/httpcore-4.4.13.jar:lib/httpclient-4.5.13.jar:lib/httpasyncclient-4.1.5.jar:lib/gson-2.9.0.jar:lib/commons-logging-1.2.jar:lib/commons-codec-1.15.jar:lib/py4j0.10.7.jar:lib/junit-4.12.jar:lib/hamcrest-core-1.3.jar:lib/commons-cli-1.3.1.jar:lib/slf4j-api-1.7.25.jar:lib/com.anylogic.engine.jar:lib/com.anylogic.engine.nl.jar:lib/com.anylogic.engine.sa.jar:lib/sa/executor-basic-8.3.jar:lib/sa/ioutil-8.3.jar:lib/sa/com.anylogic.engine.sa.web.jar:lib/sa/util-8.3.jar:lib/sa/spark/spark-core-2.9.3.jar:lib/sa/spark/jetty-continuation-9.4.31.v20200723.jar:lib/sa/spark/jetty-security-9.4.31.v20200723.jar:lib/sa/spark/websocket-common-9.4.31.v20200723.jar:lib/sa/spark/jetty-io-9.4.31.v20200723.jar:lib/sa/spark/websocket-client-9.4.31.v20200723.jar:lib/sa/spark/websocket-server-9.4.31.v20200723.jar:lib/sa/spark/jetty-servlets-9.4.31.v20200723.jar:lib/sa/spark/javax.servlet-api-3.1.0.jar:lib/sa/spark/jetty-client-9.4.31.v20200723.jar:lib/sa/spark/slf4j-api-1.7.25.jar:lib/sa/spark/jetty-webapp-9.4.31.v20200723.jar:lib/sa/spark/jetty-http-9.4.31.v20200723.jar:lib/sa/spark/jetty-util-9.4.31.v20200723.jar:lib/sa/spark/jetty-xml-9.4.31.v20200723.jar:lib/sa/spark/websocket-servlet-9.4.31.v20200723.jar:lib/sa/spark/websocket-api-9.4.31.v20200723.jar:lib/sa/spark/jetty-server-9.4.31.v20200723.jar:lib/sa/spark/jetty-servlet-9.4.31.v20200723.jar:lib/sa/jackson/jackson-databind-2.12.2.jar:lib/sa/jackson/jackson-annotations-2.12.2.jar:lib/sa/jackson/jackson-core-2.12.2.jar:lib/database/querydsl/querydsl-core-4.2.1.jar:lib/database/querydsl/querydsl-sql-4.2.1.jar:lib/database/querydsl/querydsl-sql-codegen-4.2.1.jar:lib/database/querydsl/guava-18.0.jar -Xmx128m com.alpyperl.examples.cartpole_v1.Simulation $*

cd "$DIR_BACKUP_XJAL"
