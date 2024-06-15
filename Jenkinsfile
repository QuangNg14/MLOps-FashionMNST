pipeline {
    agent {
        docker {
            image 'mrsnoo123/jenkins-agent-aws-dvc:latest'
            args '-u root:root' // Run as root user to avoid permission issues
        }
    }

    environment {
        AWS_DEFAULT_REGION = 'us-west-2'
    }

    stages {
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
