# Wine Quality Prediction
## Steps to set up and run this project without docker-
1. Login to aws  
2. Create EMR cluster to train model in parallel  
3. Create S3 bucket  
4. Write python code to train model in WinePrediction.py file    
5. Upload the python file and TrainingDataset.csv to S3 bucket  
6. Connect to EMR cluster using SSH connection with key  
7. Update the EMR cluster using sudo yum update  
8. Using "aws s3 cp s3://raghuvp/WinePrediction.py ./" command download the S3 object to EMR  
9. Run the WinePrediction.py file using command "spark-submit WinePrediction.py"  
10. Terminate the EMR cluster  
11. Create new EC2 instance    
12. Write python code for validation of model in WineValidation.py file  
13. Upload the WineValidation.py file and ValidationDataset.csv to S3 bucket  
14. Connect to EC2 instance using SSH connection with key  
15. Update the instance  
16. Download and install java on EC2 instance  
17. Download and install Apache Spark on EC2 instance  
18. Set the environment variables for both java and spark  
19. Using "aws s3 cp s3://raghuvp/WinePrediction.py ./" command download the S3 object to EC2 instance  
20. Run WineValidation.py file using command "spark-submit WineValidation.py"  
21. Terminate the EC2 instance  
