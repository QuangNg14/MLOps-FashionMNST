pipeline {
    agent any

    environment {
        AWS_DEFAULT_REGION = 'us-west-2'
    }

    stages {
        // stage('Install AWS CLI') {
        //     steps {
        //         sh '''
        //             if ! command -v aws &> /dev/null; then
        //                 echo "AWS CLI not found, installing..."
        //                 if [[ "$OSTYPE" == "darwin"* ]]; then
        //                     curl "https://awscli.amazonaws.com/AWSCLIV2.pkg" -o "AWSCLIV2.pkg"
        //                     sudo installer -pkg AWSCLIV2.pkg -target /
        //                 elif [[ "$OSTYPE" == "linux-gnu"* ]]; then
        //                     curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"
        //                     unzip awscliv2.zip
        //                     sudo ./aws/install
        //                 fi
        //             fi
        //             aws --version
        //         '''
        //     }
        // }

        stage('Checkout Repository') {
            steps {
                withCredentials([string(credentialsId: 'github-token', variable: 'GITHUB_TOKEN')]) {
                    git url: 'https://github.com/QuangNg14/MLOps-FashionMNST', branch: 'main', credentialsId: 'github-token'
                }
            }
        }

        stage('Set up Python') {
            steps {
                sh '''
                    python3 -m pip install --upgrade pip
                    pip3 install -r requirements.txt
                '''
            }
        }

        // stage('Configure AWS credentials') {
        //     steps {
        //         withCredentials([string(credentialsId: 'aws-access-key-id', variable: 'AWS_ACCESS_KEY_ID'), 
        //                          string(credentialsId: 'aws-secret-access-key', variable: 'AWS_SECRET_ACCESS_KEY')]) {
        //             sh '''
        //                 aws configure set aws_access_key_id $AWS_ACCESS_KEY_ID
        //                 aws configure set aws_secret_access_key $AWS_SECRET_ACCESS_KEY
        //                 aws configure set default.region ${AWS_DEFAULT_REGION}
        //             '''
        //         }
        //     }
        // }

        // stage('Other Stage') {
        //     steps {
        //         echo "AWS Access Key: ${env.AWS_ACCESS_KEY_ID}"
        //         echo "AWS Secret Key: ${env.AWS_SECRET_ACCESS_KEY}"
        //         echo "AWS Region: ${env.AWS_DEFAULT_REGION}"
        //     }
        // }

        stage('Run training') {
            steps {
                sh '''
                    python3 app.py
                    dvc push
                '''
            }
        }
    }

    post {
        always {
            cleanWs()
        }
        success {
            echo 'Training completed successfully!'
        }
        failure {
            echo 'Training failed. Please check the logs.'
        }
    }
}
