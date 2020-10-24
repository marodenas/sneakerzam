# sneakerzam
SneakerZam is a python algorithm that predict sneaker's model

# SneakerZam
## Final Proyect - Ironhack
  
![Image](https://images.unsplash.com/photo-1551901460-c84042b6e4ae?ixlib=rb-1.2.1&auto=format&fit=crop&w=1920&h=500&q=80)  

## Table of Contents  

* [About the Project](#about-the-project)  
  * [Challenge 1](#pushpin-challenge-1)  
  * [Challenge 3](#pushpin-challenge-2)  
  * [Challenge 2](#pushpin-challenge-3)      
  * [Built With](#hammer-built-with)  
* [How to use the pipeline](#how-to-use-the-pipeline)  
  * [Prerequisites](#page_with_curl-prerequisites)  
  * [Inputs](#computer-inputs)  
  * [Folder Structure](#file_folder-folder-structure)  
 * [Procesing Stages](#procesing-stages)
	  * [Acquistion](#electric_plug-acquisition)  
	  * [Wrangling](#wrench-wrangling)  
	  * [Analysis](#rocket-analysis)  
	  * [Reporting](#mailbox-reporting)  
* [Next Stages](#next-stages) 
  
## About the project  
  
  
###  :pushpin: Part 1  
  

  
###  :pushpin: Part 2  
  

  
  
###  :pushpin: Part 3  

###  :pushpin: Part 4  
  



  ###  :hammer: Built With   
The core of the project is Python 3.7.3, but you have to install those libraries for run the script.   
Native packages:  
- [Argparse](https://docs.python.org/3.7/library/argparse.html)  
- [Configparser](https://docs.python.org/3/library/configparser.html)  
- [Datetime](https://docs.python.org/2/library/datetime.html)  
- [Re](https://docs.python.org/3/library/re.html)  
- [Smtplib](https://docs.python.org/3/library/smtplib.html)  
- [Email](https://docs.python.org/3/library/email.examples.html)  
  
Furthermore, it is has to be installed the following libraries:  
- [SQL Alchemy (v.1.3.17)](https://docs.sqlalchemy.org/en/13/intro.html)  
- [Pandas (v.0.24.2)](https://pandas.pydata.org/pandas-docs/stable/reference/index.html)  
- [Numpy (v.1.18.1)](https://numpy.org/doc/stable/)  
- [Requests (v.2.23.0)](https://requests.readthedocs.io/)  
- [Beautiful Soup (v.4.9.1)](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)  
- [Pysftp (0.2.9) ](https://pypi.org/project/pysftp/)  

  
## **How to use the app**
###  **:page_with_curl:Prerequisites**  
Please, install all the libraries mentinoned in [Built With](#built-with) in your enviroment in order to run the script.  
   
Furthemore, there is a config.ini file where you have to specify all parameters that you need to run the script. Please, fill out all the variables in order to run the script. You will need the database, api url, and webscraping url.   
  
In order to send an email and upload the outcome csv, you will need to provide a gmail email, reciever, pass.
For uploading the html version of the results, you will need to provide the sftp user and password of your hosting. 

Don't worry about privacity, this config file will not be uploaded to github.  

Example of config.ini file: 

```  
[email]  
user = your user email  
password = your password email  
receiver = people who will recieve the email  
  
[data]  
db = database path  
url = url for scrapping country codes  
  
[website]  
myHostname = xxxxxxx  
myUsername = xxxxxxx  
myPassword = xxxxxxx ```  
```
  

  
### **:computer: Inputs**  
 
  
### :file_folder: **Folder structure**  
```
└── ih_datamadpt0420_project_m1  
    ├── __trash__  
  ├── .gitignore  
    ├── .env  
    ├── requeriments.txt  
    ├── README.md  
    ├── main_script.py  
    ├── config.ini  
    ├── notebooks  
    │   ├── notebook1.ipynb  
    │   └── notebook2.ipynb  
    ├── p_acquisition  
    │   ├── __init__.py  
    │   └── m_acquisition.py  
    ├── p_analysis  
    │   ├── __init__.py  
    │   └── m_analysis.py  
    ├── p_reporting  
    │   ├── __init__.py  
    │   └── m_reporting.py  
    ├── p_wrangling  
    │   ├── __init__.py  
    │   └── m_wrangling.py  
    └── data  
        ├── html  
        ├── raw  
        └── results  
```  
  

## **Processing stages**  
  
### **:electric_plug: Acquisition**  
  

  
### **:wrench: Wrangling**  



 ### **:rocket: Analysis**  


  
 ### **:mailbox: Reporting**  
 

  
 ---  
### ** Next stages**  

