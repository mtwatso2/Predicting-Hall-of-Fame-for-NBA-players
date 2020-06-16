# -*- coding: utf-8 -*-
"""
Created on Thu Oct  3 17:07:15 2019

@author: MWatson717

DSC 478: Final Project
"""

import pandas as pd

data = pd.read_csv("nba.csv")

data.info()
data.describe(include = 'all')
#should only be 500 observations

data.tail()
data.drop(data.tail(1).index,inplace=True)
data.loc[282, "Player"] = "Eddie Johnson 2"  

data.info()

#67 missing for steal and block, 121 missing for 3pt% (stats werent recorded, fill with median by position)

pg = ["Oscar Robertson", "Magic Johnson", "Jerry West", "Isiah Thomas", "John Stockton", "Bob Cousy", "Jason Kidd", "Walt Frazier", "Gary Payton", 
      "Allen Iverson", "Earl Monroe", "Steve Nash", "Tiny Archibald", "Dennis Johnson", "Dave Bing", "Hal Greer", "Lenny Wilkens", "Joe Dumars",
      "Tim Hardaway", "Sidney Moncrief", "Slater Martin","Kevin Johnson", "Chris Paul", "Norm Van Lier", "Jo Jo White", "Dick McGuire",
      "Maurice Cheeks", "Chauncey Billups", "Deron Williams", "Guy Rodgers", "Phil Chenier", "Bob Davies", "Tony Parker", "Paul Westphal",
      "Gus Williams", "Gene Shue", "Mark Price", "Calvin Murphy", "Mark Jackson", "Terry Porter", "Reggie Theus", "K.C. Jones",
      "Gilbert Arenas", "Norm Nixon", "Sam Cassell", "Stephon Marbury", "Fat Lever", "Baron Davis", "Mookie Blaylock", "Rod Strickland",
      "Derek Harper", "Steve Francis", "Quinn Buckner", "Mike Bibby", "Andre Miller", "Geoff Petrie", "John Lucas", "Terrell Brandon",
      "Sleepy Floyd", "Doc Rivers", "Larry Costello", "Phil Ford", "Nick Van Exel", "Lucius Allen", "Damon Stoudamire", "Slick Watts",
      "Kenny Anderson", "Ray Williams", "Lionel Hollins", "Kenny Smith", "Michael Adams", "Jameer Nelson", "Muggsy Bogues", "Mahmoud Abdul-Rauf",
      "Sherman Douglas", "Walt Hazzard", "Scott Skiles", "Flynn Robinson", "Derek Fisher", "Larry Drew", "Vern Fleming", "Johnny Dawkins",
      "Pooh Richardson", "Nate McMillan", "Ernie DiGregorio", "Al Attles", "B.J. Armstrong", "Spud Webb", "Kelvin Ransey", "Eric Money",
      "Jason Williams", "Devin Harris", "Jay Humphries", "Don Buse", "Wali Jones", "Dana Barros", "Avery Johnson"] 

sg = ["Michael Jordan", "Kobe Bryant", "George Gervin", "Dwyane Wade", "Reggie Miller", "Pete Maravich", "David Thompson", "Sam Jones", 
      "Bill Sharman", "Ray Allen", "Mitch Richmond", "Grant Hill", "Vince Carter", "Kevin Porter", "Richie Guerin", "Marques Johnson", 
      "Anfernee Hardaway", "Charlie Scott", "Michael Cooper", "Phil Smith", "Manu Ginóbili", "Bobby Wanzer", "Otis Birdsong", "Randy Smith", 
      "Rolando Blackman", "Andy Phillip", "Andrew Toney", "Micheal Ray Richardson", "Archie Clark", "Jimmy Walker", "Doug Collins", 
      "Alvin Robertson", "World B. Free", "Don Ohl", "Carl Braun", "Jeff Malone", "John Williamson", "Darrell Griffith", "Dale Ellis",
      "Michael Redd", "Austin Carr", "Max Zaslofsky", "Doug Christie", "Eddie Jones", "Jeff Hornacek", "Brian Winters", "Eddie Johnson",
      "Jeff Mullins", "Jerry Stackhouse", "Don Chaney", "Fred Brown", "Dick Barnett", "Dick Van Arsdale", "Danny Ainge", "Fred Carter",
      "Allan Houston", "Byron Scott", "Ron Harper", "Ben Gordon", "Vinnie Johnson", "Nick Anderson", "Paul Pressey", "Monta Ellis",
      "Steve Smith", "Ricky Sobers", "Ricky Pierce", "John Starks", "Jason Terry", "Hersey Hawkins", "Mike Newlin", "Kevin Loughery",
      "Dick Garmaker", "Vernon Maxwell", "Mike Woodson", "Johnny Davis", "Bucky Bockhorn", "Eddie Miles", "Herm Gilliam", "Adrian Smith",
      "Rex Chapman", "Jamal Crawford", "Quintin Dailey", "Jim Paxson", "Gerald Wilkins", "Dell Curry", "Kevin Martin", "Kerry Kittles",
      "Kendall Gill", "Larry Hughes", "Jon McGlocklin", "Clem Haskins", "John Long", "Šarūnas Marčiulionis", "Cuttino Mobley", "David Wesley"]

sf = ["Elgin Baylor", "Julius Erving", "John Havlicek", "Rick Barry", "Scottie Pippen", "LeBron James", "Billy Cunningham", "Clyde Drexler", 
      "Dominique Wilkins", "Dennis Rodman", "James Worthy", "Jack Twyman", "Gus Johnson", "Tom Heinsohn", "Bernard King", "Paul Arizin",
      "Dave DeBusschere", "Paul Pierce", "Alex English", "Adrian Dantley", "Cliff Hagan", "Chris Mullin", "Bob Dandridge", "Bob Love",
      "Jim Pollard", "Tracy McGrady", "Walter Davis", "Carmelo Anthony", "Jamaal Wilkes", "Chet Walker", "George Yardley", "Willie Naulls",
      "Lou Hudson", "Paul Silas", "Metta World Peace", "Shawn Marion", "Tom Gola", "Bill Bradley", "Larry Johnson", "Jerry Sloan",
      "Jamal Mashburn", "Joe Caldwell", "Mark Aguirre", "Kiki Vandeweghe", "Lamar Odom", "Kelly Tripucka", "Cedric Maxwell", "Reggie Lewis",
      "Richard Hamilton", "Peja Stojaković", "Latrell Sprewell", "Jason Richardson", "Purvis Short", "Antawn Jamison", "Jim McMillian",
      "Glenn Robinson", "Bruce Bowen", "Detlef Schrempf", "Terry Dischinger", "Toni Kukoč", "Frank Ramsey", "Anthony Mason", "Happy Hairston",
      "Shareef Abdur-Rahim", "Dan Majerle", "Jack Marin", "Clifford Robinson", "Scott Wedman", "Xavier McDaniel", "Eddie Johnson 2",
      "Cliff Robinson", "Michael Finley", "Mike Mitchell", "Glen Rice", "Calvin Natt", "Rashard Lewis", "Tom Sanders", "Sean Elliott",
      "Tom Van Arsdale", "Gerald Wallace", "Caron Butler", "Lionel Simmons", "Greg Ballard", "Danny Granger", "Andre Iguodala", "Billy Knight",
      "Jay Vincent", "Jim Jackson", "Rodney McCray", "Campy Russell", "Tayshaun Prince", "Chuck Person", "John Johnson", "Jalen Rose",
      "Mike Bantom", "Stephen Jackson", "Richard Jefferson", "Corey Maggette", "Junior Bridgeman", "M.L. Carr", "Quentin Richardson",
      "Derrick McKey", "Keith Van Horn", "Cedric Ceballos", "Michael Brooks", "Ken Norman", "Don Kojis", "Luol Deng", "Jerome Kersey",
      "Don Nelson", "Mike Miller", "Roy Hinson", "Andrei Kirilenko", "Shane Battier", "Albert King", "Josh Howard", "Hedo Türkoğlu",
      "Wally Szczerbiak", "Billy Owens", "Phil Hubbard", "Robert Reid", "Gene Banks", "Reggie Williams"]

pf = ["Larry Bird", "Bob Pettit", "Moses Malone", "Karl Malone", "Charles Barkley", "Elvin Hayes", "Kevin McHale", "Kevin Garnett", 
      "Willis Reed", "Wes Unseld", "Nate Thurmond", "Dolph Schayes", "Jerry Lucas", "Dave Cowens", "Bob McAdoo", "Dirk Nowitzki",
      "Connie Hawkins", "Joe Fulks", "Alonzo Mourning", "Chris Webber", "Dwight Howard", "Spencer Haywood", "Buck Williams",
      "Dan Issel", "George McGinnis", "Zelmo Beaty", "Tom Chambers", "Ben Wallace", "Bailey Howell", "Rudy Tomjanovich",
      "Maurice Lucas", "Amar'e Stoudemire", "Vern Mikkelsen", "Rudy LaRusso", "Terry Cummings", "Shawn Kemp", "Horace Grant",
      "Rasheed Wallace", "Bobby Jones", "Johnny Green", "John Drew", "Larry Nance", "Clyde Lovellette", "Chris Bosh", "Jermaine O'Neal",
      "Elton Brand", "Derrick Coleman", "Joe Johnson", "Larry Kenon", "Bill Bridges", "Alvan Adams", "Carlos Boozer", "Marcus Camby",
      "Danny Manning", "Charles Oakley", "Sidney Wicks", "Ray Scott", "Cazzie Russell", "Mickey Johnson", "Orlando Woolridge",
      "Clark Kellogg", "Antoine Walker", "Truck Robinson", "A.C. Green", "Ralph Sampson", "David West", "Dan Roundfield", "Otis Thorpe",
      "Zach Randolph", "Bob Boozer", "Kenny Sears", "Tom Gugliotta", "Kenyon Martin", "Sam Perkins", "Tom Meschery", "Wayman Tisdale",
      "Luke Jackson", "Kevin Willis", "Roy Tarpley", "Vin Baker", "Nat Clifton", "Robert Horry", "Armen Gilliam", "John Shumate",
      "Antonio McDyess", "Juwan Howard", "Lloyd Neal", "Lonnie Shelton", "Larry Smith", "Woody Sauldsberry", "Jim Washington",
      "Kermit Washington", "Antonio Davis", "Christian Laettner", "Thurl Bailey", "Al Harrington", "Kenny Carr", "Ron Behagen",
      "Brian Grant", "Tyrone Hill", "Troy Murphy", "Curtis Rowe", "Dave Greenwood", "Charles Smith", "David Lee", "Grant Long",
      "LaPhonso Ellis", "Gar Heard", "Terry Catledge", "Bob Kauffman", "Loy Vaught", "Mitch Kupchak", "Joe Graboski", "Antoine Carr",
      "Joe Smith", "P.J. Brown", "Drew Gooden"]

c = ["Wilt Chamberlain", "Bill Russell", "Shaquille O'Neal", "Kareem Abdul-Jabbar", "Tim Duncan", "Hakeem Olajuwon", "David Robinson", 
     "George Mikan", "Patrick Ewing", "Walt Bellamy", "Bob Lanier", "Robert Parish", "Bill Walton", "Neil Johnston", "Ed Macauley",
     "Artis Gilmore", "Dikembe Mutombo", "Pau Gasol", "Brad Daugherty", "Jack Sikma", "Harry Gallatin", "Larry Foust", "Red Kerr",
     "Yao Ming", "Bob Rule", "Wayne Embry", "Arnie Risen", "Bill Laimbeer", "Jeff Ruland", "Walter Dukes", "Zydrunas Ilgauskas",
     "Rony Seikaly", "Caldwell Jones", "Rik Smits", "Vlade Divac", "Bill Cartwright", "Mel Hutchins", "Swen Nater", "Sam Lacey",
     "Elmore Smith", "Joe Barry Carroll", "Theo Ratliff", "Arvydas Sabonis", "Jim Chones", "Darryl Dawkins", "Al Jefferson",
     "Steve Stipanovich", "Mark Eaton", "Mychal Thompson", "Ray Felix", "Steve Johnson", "Michael Cage", "Dale Davis", "Andrew Bogut",
     "James Edwards", "Hot Rod Williams", "Sam Bowie", "Zaid Abdul-Aziz", "Emeka Okafor", "Tree Rollins", "Marvin Webster",
     "Elden Campbell", "Clyde Lee", "Clifford Ray", "Manute Bol", "Kevin Duckworth", "Chris Kaman", "Pervis Ellison"]

def pos(row):
    if row['Player'] in pg:
        val = "PG"
    elif row['Player'] in sg:
        val = "SG"
    elif row['Player'] in sf:
        val = "SF"
    elif row['Player'] in pf:
        val = "PF"
    elif row['Player'] in c:
        val = "C"
    else:
        val = ""
    return val

data['Position'] = data.apply(pos, axis = 1)

data.tail()

data.loc[499, "Position"] = "C"

data.tail()

data['STL'] = data.groupby('Position')['STL'].transform(lambda x: x.fillna(x.median()))

data['BLK'] = data.groupby('Position')['BLK'].transform(lambda x: x.fillna(x.median()))

data['3P%'] = data.groupby('Position')['3P%'].transform(lambda x: x.fillna(x.median()))

data.info()

#https://www.landofbasketball.com/hall_of_fame/hall_of_famers_by_year.htm

hof = ["George Mikan", "Ed Macauley", "Andy Phillip", "Bob Davies", "Bob Pettit", "Bob Cousy", "Dolph Schayes", "Bill Russell", "Tom Gola",
       "Bill Sharman", "Elgin Baylor", "Paul Arizin", "Joe Fulks", "Cliff Hagan", "Jim Pollard", "Wilt Chamberlain", "Jerry Lucas", 
       "Oscar Robertson", "Jerry West", "Hal Greer", "Slater Martin", "Frank Ramsey", "Willis Reed", "Bill Bradley", "Dave DeBusschere",
       "Jack Twyman", "John Havlicek", "Sam Jones", "Al Cervi", "Nate Thurmond", "Billy Cunningham", "Tom Heinsohn", "Rick Barry",
       "Walt Frazier", "Bob Houbregs", "Pete Maravich", "Bobby Wanzer", "Clyde Lovellette", "Wes Unseld", "K.C. Jones", "Lenny Wilkins",
       "Dave Bing", "Dave Cowens", "Harry Gallatin", "Connie Hawkins", "Bob Lanier", "Walt Bellamy", "Julius Erving", "Dan Issel",
       "Dick McGuire", "Calvin Murphy", "Bill Walton", "Buddy Jeannette", "Kareem Abdul-Jabbar", "Vern Mikkelsen", "George Gervin",
       "Gail Goodrich", "David Thompson", "George Yardley", "Alex English", "Bailey Howell", "Larry Bird", "Arnie Risen", "Wayne Embry",
       "Kevin McHale", "Bob McAdoo", "Isiah Thomas", "Moses Malone", "Magic Johnson", "Drazen Petrovic", "Earl Lloyd", "Robert Parish",
       "James Worthy", "Clyde Drexler", "Maurice Stokes", "Charles Barkley", "Joe Dumars", "Dominique Wilkins", "Adrian Dantley", 
       "Patrick Ewing", "Hakeem Olajuwon", "Michael Jordan", "David Robinson", "John Stockton", "Dennis Johnson", "Gus Johnson",
       "Karl Malone", "Scottie Pippen", "Artis Gilmore", "Chris Mullin", "Dennis Rodman", "Arvydas Sabonis", "Tom Sanders", "Don Barksdale",
       "Mel Daniels", "Reggie Miller", "Ralph Sampson", "Chet Walker", "Jamaal Wilkes", "Richie Guerin", "Bernard King", "Gary Payton",
       "Nathaniel Clifton", "Slick Leonard", "Sarunas Marciulionis", "Alonso Mourning", "Mitch Richmond", "Guy Rodgers", "Louis Dampier",
       "Spencer Haywood", "Tom Heinsohn", "Dikembe Mutombo", "Jo Jo White", "Zelmo Beaty", "Allen Iverson", "Yao Ming", "Shaquille O'Neal",
       "George McGinnis", "Tracy McGrady", "Ray Allen", "Maruice Cheeks", "Grant Hill", "Jason Kidd", "Steve Nash", "Dino Radja", 
       "Charlie Scott", "Rod Thorn", "Carl Braun", "Chuck Cooper", "Vlade Divac", "Bobby Jones", "Sidney Moncrief", "Jack Sikma", "Paul Wetphal"]

def hf(row):
    if row['Player'] in hof:
        val = 1
    else:
        val = 0 
    return val

data['HoF'] = data.apply(hf, axis = 1)

data['HoF'] = pd.Categorical(data.HoF)

print(data.dtypes)

data['HoF'].value_counts()

len(hof)

#134 hall of fame players, 113 are in top 500

data['Position'].value_counts()

data['Years Played'] = data['To'] - data['From']


from sklearn import model_selection

nba = data.to_csv('nba.csv')

#Box plots for HoF variable: EDA

data_sub = data[data["To"] > 2015]

data_naoi = data[~data.isin(data_sub)].dropna()

data.describe(include = 'all')

import matplotlib.pyplot as plt

data_naoi.boxplot(column = 'Rank', by = 'HoF')
plt.title("Boxplot of Rank by HoF")
plt.suptitle("")
plt.xlabel("Hall of Fame")
plt.ylabel("Rank")
plt.show()

data_naoi.boxplot(column = 'PTS', by = 'HoF')
plt.title("Boxplot of Points Per Game by HoF")
plt.suptitle("")
plt.xlabel("Hall of Fame")
plt.ylabel("Points Per Game")
plt.show()

data_naoi.boxplot(column = 'WS', by = 'HoF')
plt.title("Boxplot of Total Win Shares by HoF")
plt.suptitle("")
plt.xlabel("Hall of Fame")
plt.ylabel("Total Win Shares")
plt.show()

data_naoi.boxplot(column = 'WS/48', by = 'HoF')
plt.title("Boxplot of Win Shares per 48 Minutes by HoF")
plt.suptitle("")
plt.xlabel("Hall of Fame")
plt.ylabel("Win Shares Per 48 minutes (game)")
plt.show()

data_naoi.boxplot(column = 'Years Played', by = 'HoF')
plt.title("Boxplot of Total Years Played by HoF")
plt.suptitle("")
plt.xlabel("Hall of Fame")
plt.ylabel("Total Years Played")
plt.show()


#Logistic Regression
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

validate = data_sub.drop(['HoF', 'Position'], axis = 1)

y = data_naoi['HoF']

data_naoi = data_naoi.drop(['HoF', 'Position', 'Player'], axis = 1)

nba_train, nba_test, hof_train, hof_test = train_test_split(data_naoi, y, test_size=0.2, random_state=33)

lr = LogisticRegression()

lr.fit(nba_train, hof_train)

print(lr.coef_)
print(lr.intercept_)

col = list(data_naoi.columns)
coef = lr.coef_
for i in range(len(col)):
    print("{}: {:0.3f}".format(col[i], coef[0, i]))

y_pred = lr.predict(nba_test)

lr_cm = confusion_matrix(hof_test, y_pred)
print(lr_cm)

acc = (lr_cm[0,0] + lr_cm[1,1]) / (lr_cm[0,0] + lr_cm[0,1] + lr_cm[1,0]+ lr_cm[1,1])
print("The Overall Accuracy is: {}%".format(acc*100))

players = list(validate['Player'])
validate_nn = validate.drop('Player', axis = 1)
probs = lr.predict_proba(validate_nn)

print("Probability of joining hall of fame: ")
for i in range(len(validate_nn)):
    print("{}: {:0.3f}%".format(players[i], probs[i, 1]*100))
    


#KNN:

from sklearn import preprocessing, neighbors, decomposition

import numpy as np


data_norm = data.drop(['Player', 'HoF'], axis = 1)

data_norm = pd.get_dummies(data_norm, columns = ['Position'])

min_max_scaler = preprocessing.MinMaxScaler().fit(data_norm)
data_norm = min_max_scaler.transform(data_norm)

row_ix = np.array([7, 9, 29, 30, 48, 54, 76, 85, 95, 98, 104, 106, 127, 138, 139, 142, 143, 170, 176, 183, 254, 
                   260, 272, 323, 332, 345, 346, 352, 356, 375, 379, 413, 417, 430, 439, 451, 454, 457, 460, 494, 496])
    
col_ix = np.array([0, 1, 1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20])

val = data_norm[row_ix[:, None], col_ix]

data_norm = np.delete(data_norm, row_ix, 0)

nba_train, nba_test, hof_train, hof_test = train_test_split(data_norm, y, test_size=0.2, random_state=33)


def kNN(x_train, y_train, x_test, y_test):
    for i in range(1,11):
        knnclf = neighbors.KNeighborsClassifier(i)
        knnclf.fit(x_train, y_train)
        knnpreds_test = knnclf.predict(x_test)
        print("Results for {} neighbors: ".format(i))
        print("Classification Report:")
        print(classification_report(y_test, knnpreds_test))
        print("Confusion Matrix:")
        print(confusion_matrix(y_test, knnpreds_test))
        print("\n")
        
kNN(nba_train, hof_train, nba_test, hof_test)

#it looks like a k of 5 gives us the best results

knnclf = neighbors.KNeighborsClassifier(5)
knnclf.fit(nba_train, hof_train)

knnpreds_test = knnclf.predict(nba_test)
knncm = confusion_matrix(hof_test, knnpreds_test)
print(knncm)
print ("Score on Training: ", knnclf.score(nba_train, hof_train))
print ("Score on Test: ", knnclf.score(nba_test, hof_test))


cv_scores = model_selection.cross_val_score(knnclf, nba_train, hof_train, cv = 5)
print(cv_scores)
print("Overall Accuracy on X-Val: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))

probs = knnclf.predict_proba(val)
print("Probability of joining hall of fame: ")
for i in range(len(validate_nn)):
    print("{}: {}%".format(players[i], probs[i, 1]*100))



#%matplotlib inline
plt.matshow(knncm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()


#PCA for LR
pca = decomposition.PCA(n_components = 10)

pca.fit_transform(data_naoi)

var = pca.explained_variance_ratio_
print(var)
var[0] + var[1]

pca = decomposition.PCA(n_components = 2)
naoi2 = pca.fit_transform(data_naoi)
validate_nn2 = pca.fit_transform(validate_nn)


#PCA for KNN

pca = decomposition.PCA(n_components = 10)

pca.fit_transform(data_norm)

var = pca.explained_variance_ratio_
print(var)
sum(var)

pca = decomposition.PCA(n_components = 10)
data2 = pca.fit_transform(data_norm)   
val2 = pca.fit_transform(val)
 

#Rerun LR on PCA Data
nba_train, nba_test, hof_train, hof_test = train_test_split(naoi2, y, test_size=0.2, random_state=33)

lr = LogisticRegression()

lr.fit(nba_train, hof_train)

coef = lr.coef_

for i in range(2):
    print("Component {}: {:0.3f}".format(i, coef[0, i]))
print("\n")
print(lr.intercept_)

y_pred = lr.predict(nba_test)

lr_cm = confusion_matrix(hof_test, y_pred)
print(lr_cm)

acc = (lr_cm[0,0] + lr_cm[1,1]) / (lr_cm[0,0] + lr_cm[0,1] + lr_cm[1,0]+ lr_cm[1,1])
print("The Overall Accuracy is: {:0.2f}%".format(acc*100))

probs = lr.predict_proba(validate_nn2)

print("Probability of joining hall of fame: ")
for i in range(len(validate_nn)):
    print("{}: {:0.3f}%".format(players[i], probs[i, 1]*100))


#rerun KNN on PCA data
nba_train, nba_test, hof_train, hof_test = train_test_split(data2, y, test_size=0.2, random_state=33)

kNN(nba_train, hof_train, nba_test, hof_test)

#it looks like k of 1 performed best, only miss classifying 6 out of 92

knnclf = neighbors.KNeighborsClassifier(2)
knnclf.fit(nba_train, hof_train)

knnpreds_test = knnclf.predict(nba_test)
knncm = confusion_matrix(hof_test, knnpreds_test)
print(knncm)

print ("Score on Training: ", knnclf.score(nba_train, hof_train))
print ("Score on Test: ", knnclf.score(nba_test, hof_test))

#%matplotlib inline
plt.matshow(knncm)
plt.title('Confusion matrix')
plt.colorbar()
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.show()

#overall, dimenstionality reduction gave us slightly worse results, even though the two component acount for over 98% of the variance

cv_scores = model_selection.cross_val_score(knnclf, nba_train, hof_train, cv = 5)
print(cv_scores)
print("Overall Accuracy on X-Val: %0.2f (+/- %0.2f)" % (cv_scores.mean(), cv_scores.std() * 2))

#CV scores are about the same

probs = knnclf.predict_proba(val2)
print("Probability of joining hall of fame: ")
for i in range(len(validate_nn)):
    print("{}: {}%".format(players[i], probs[i, 1]*100))