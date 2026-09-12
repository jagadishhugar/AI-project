```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pandas import get_dummies
from sklearn.cluster import k_means
from webencodings import encode

```


```python
df=pd.read_csv("students-data-50.1.csv")
```


```python

```


```python
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Attendance (%)</th>
      <th>Study_Hours</th>
      <th>Previous_Marks</th>
      <th>Assignments_Completed</th>
      <th>Join_Date</th>
      <th>City</th>
      <th>Extra_Classes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rahul</td>
      <td>M</td>
      <td>21</td>
      <td>92</td>
      <td>3.5</td>
      <td>78</td>
      <td>9</td>
      <td>2024-06-10</td>
      <td>Bengaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Priya</td>
      <td>Female</td>
      <td>22</td>
      <td>88%</td>
      <td>4.0</td>
      <td>85</td>
      <td>10</td>
      <td>10/06/2024</td>
      <td>Bangalore</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Amit</td>
      <td>MALE</td>
      <td>20</td>
      <td>76</td>
      <td>2.5</td>
      <td>67</td>
      <td>8</td>
      <td>2024/06/12</td>
      <td>Mysuru</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sneha</td>
      <td>F</td>
      <td>23</td>
      <td>95</td>
      <td>5.0</td>
      <td>91</td>
      <td>10</td>
      <td>12-06-2024</td>
      <td>Mysore</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kiran</td>
      <td>Male</td>
      <td>21</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>72</td>
      <td>7</td>
      <td>2024-06-15</td>
      <td>Hubballi</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Anjali</td>
      <td>female</td>
      <td>20</td>
      <td>89</td>
      <td>4.5</td>
      <td>88</td>
      <td>9</td>
      <td>15/06/2024</td>
      <td>Hubli</td>
      <td>No</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Vikram</td>
      <td>M</td>
      <td>24</td>
      <td>72</td>
      <td>2.0</td>
      <td>61</td>
      <td>6</td>
      <td>2024-06-18</td>
      <td>Dharwad</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Neha</td>
      <td>Female</td>
      <td>22</td>
      <td>91</td>
      <td>NaN</td>
      <td>93</td>
      <td>10</td>
      <td>18/06/2024</td>
      <td>Dharwad</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Rohan</td>
      <td>Male</td>
      <td>19</td>
      <td>68</td>
      <td>1.5</td>
      <td>55</td>
      <td>5</td>
      <td>2024/06/20</td>
      <td>Belagavi</td>
      <td>No</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Divya</td>
      <td>FEMALE</td>
      <td>21</td>
      <td>84</td>
      <td>3.0</td>
      <td>76</td>
      <td>8</td>
      <td>20-06-2024</td>
      <td>Belgaum</td>
      <td>N</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.head(50)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Attendance (%)</th>
      <th>Study_Hours</th>
      <th>Previous_Marks</th>
      <th>Assignments_Completed</th>
      <th>Join_Date</th>
      <th>City</th>
      <th>Extra_Classes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rahul</td>
      <td>M</td>
      <td>21</td>
      <td>92</td>
      <td>3.5</td>
      <td>78</td>
      <td>9</td>
      <td>2024-06-10</td>
      <td>Bengaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Priya</td>
      <td>Female</td>
      <td>22</td>
      <td>88%</td>
      <td>4.0</td>
      <td>85</td>
      <td>10</td>
      <td>10/06/2024</td>
      <td>Bangalore</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Amit</td>
      <td>MALE</td>
      <td>20</td>
      <td>76</td>
      <td>2.5</td>
      <td>67</td>
      <td>8</td>
      <td>2024/06/12</td>
      <td>Mysuru</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sneha</td>
      <td>F</td>
      <td>23</td>
      <td>95</td>
      <td>5.0</td>
      <td>91</td>
      <td>10</td>
      <td>12-06-2024</td>
      <td>Mysore</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kiran</td>
      <td>Male</td>
      <td>21</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>72</td>
      <td>7</td>
      <td>2024-06-15</td>
      <td>Hubballi</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Anjali</td>
      <td>female</td>
      <td>20</td>
      <td>89</td>
      <td>4.5</td>
      <td>88</td>
      <td>9</td>
      <td>15/06/2024</td>
      <td>Hubli</td>
      <td>No</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Vikram</td>
      <td>M</td>
      <td>24</td>
      <td>72</td>
      <td>2.0</td>
      <td>61</td>
      <td>6</td>
      <td>2024-06-18</td>
      <td>Dharwad</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Neha</td>
      <td>Female</td>
      <td>22</td>
      <td>91</td>
      <td>NaN</td>
      <td>93</td>
      <td>10</td>
      <td>18/06/2024</td>
      <td>Dharwad</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Rohan</td>
      <td>Male</td>
      <td>19</td>
      <td>68</td>
      <td>1.5</td>
      <td>55</td>
      <td>5</td>
      <td>2024/06/20</td>
      <td>Belagavi</td>
      <td>No</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Divya</td>
      <td>FEMALE</td>
      <td>21</td>
      <td>84</td>
      <td>3.0</td>
      <td>76</td>
      <td>8</td>
      <td>20-06-2024</td>
      <td>Belgaum</td>
      <td>N</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Arjun</td>
      <td>M</td>
      <td>22</td>
      <td>79%</td>
      <td>2.5</td>
      <td>69</td>
      <td>7</td>
      <td>2024-06-22</td>
      <td>Bengaluru</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Pooja</td>
      <td>Female</td>
      <td>23</td>
      <td>96</td>
      <td>5.5</td>
      <td>95</td>
      <td>10</td>
      <td>22/06/2024</td>
      <td>Mysuru</td>
      <td>No</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Manoj</td>
      <td>Male</td>
      <td>twenty-five</td>
      <td>64</td>
      <td>1.0</td>
      <td>48</td>
      <td>4</td>
      <td>2024-06-25</td>
      <td>Tumakuru</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Asha</td>
      <td>F</td>
      <td>20</td>
      <td>87</td>
      <td>3.5</td>
      <td>82</td>
      <td>9</td>
      <td>25/06/2024</td>
      <td>Tumkur</td>
      <td>N</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Suresh</td>
      <td>male</td>
      <td>22</td>
      <td>73.5</td>
      <td>2.0</td>
      <td>64</td>
      <td>6</td>
      <td>2024/06/27</td>
      <td>Shivamogga</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Meena</td>
      <td>Female</td>
      <td>21</td>
      <td>90</td>
      <td>4.0</td>
      <td>89</td>
      <td>10</td>
      <td>27-06-2024</td>
      <td>Shimoga</td>
      <td>No</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Naveen</td>
      <td>M</td>
      <td>23</td>
      <td>78</td>
      <td>2.5</td>
      <td>71</td>
      <td>7</td>
      <td>2024-06-29</td>
      <td>Mangaluru</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kavya</td>
      <td>female</td>
      <td>20</td>
      <td>93</td>
      <td>4.5</td>
      <td>90</td>
      <td>10</td>
      <td>29/06/2024</td>
      <td>Mangalore</td>
      <td>No</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Prakash</td>
      <td>Male</td>
      <td>24</td>
      <td>69%</td>
      <td>1.5</td>
      <td>58</td>
      <td>5</td>
      <td>2024-07-01</td>
      <td>Udupi</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Lakshmi</td>
      <td>F</td>
      <td>22</td>
      <td>86%</td>
      <td>3.0</td>
      <td>81</td>
      <td>8</td>
      <td>01/07/2024</td>
      <td>Udupi</td>
      <td>N</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Harish</td>
      <td>M</td>
      <td>21</td>
      <td>82</td>
      <td>3.0</td>
      <td>74</td>
      <td>8</td>
      <td>2024/07/03</td>
      <td>Bengaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Shalini</td>
      <td>Female</td>
      <td>23</td>
      <td>94</td>
      <td>5.0</td>
      <td>92</td>
      <td>10</td>
      <td>03-07-2024</td>
      <td>Mysuru</td>
      <td>No</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Ganesh</td>
      <td>male</td>
      <td>26</td>
      <td>105</td>
      <td>0.5</td>
      <td>45</td>
      <td>3</td>
      <td>2024-07-05</td>
      <td>Dharwad</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Deepa</td>
      <td>Female</td>
      <td>21</td>
      <td>89</td>
      <td>4.0</td>
      <td>87</td>
      <td>9</td>
      <td>05/07/2024</td>
      <td>Hubballi</td>
      <td>No</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Ravi</td>
      <td>M</td>
      <td>22</td>
      <td>77</td>
      <td>2.5</td>
      <td>68</td>
      <td>seven</td>
      <td>2024-07-07</td>
      <td>Belagavi</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Swathi</td>
      <td>F</td>
      <td>20</td>
      <td>91</td>
      <td>4.5</td>
      <td>94</td>
      <td>10</td>
      <td>07/07/2024</td>
      <td>Belgaum</td>
      <td>No</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Mahesh</td>
      <td>Male</td>
      <td>23</td>
      <td>74</td>
      <td>2.0</td>
      <td>63</td>
      <td>6</td>
      <td>2024/07/09</td>
      <td>Tumakuru</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Nandini</td>
      <td>Female</td>
      <td>22</td>
      <td>88</td>
      <td>3.5</td>
      <td>85</td>
      <td>9</td>
      <td>09-07-2024</td>
      <td>Tumkur</td>
      <td>N</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Ajay</td>
      <td>M</td>
      <td>21</td>
      <td>80</td>
      <td>3.0</td>
      <td>73</td>
      <td>8</td>
      <td>2024-07-11</td>
      <td>Shivamogga</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Rekha</td>
      <td>female</td>
      <td>24</td>
      <td>97</td>
      <td>5.5</td>
      <td>96</td>
      <td>10</td>
      <td>11/07/2024</td>
      <td>Shimoga</td>
      <td>No</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Vijay</td>
      <td>Male</td>
      <td>25</td>
      <td>66</td>
      <td>1.0</td>
      <td>51</td>
      <td>4</td>
      <td>2024-07-13</td>
      <td>Mangaluru</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Sowmya</td>
      <td>F</td>
      <td>21</td>
      <td>85</td>
      <td>3.5</td>
      <td>79</td>
      <td>8</td>
      <td>13/07/2024</td>
      <td>Mangalore</td>
      <td>No</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Ramesh</td>
      <td>male</td>
      <td>22</td>
      <td>71</td>
      <td>2.0</td>
      <td>60</td>
      <td>6</td>
      <td>2024/07/15</td>
      <td>Udupi</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Bhavana</td>
      <td>Female</td>
      <td>NaN</td>
      <td>92</td>
      <td>4.0</td>
      <td>91</td>
      <td>10</td>
      <td>15-07-2024</td>
      <td>Udupi</td>
      <td>No</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Sanjay</td>
      <td>M</td>
      <td>23</td>
      <td>79</td>
      <td>2.5</td>
      <td>70</td>
      <td>7</td>
      <td>2024-07-17</td>
      <td>Bengaluru</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Aishwarya</td>
      <td>FEMALE</td>
      <td>22</td>
      <td>90</td>
      <td>4.5</td>
      <td>88</td>
      <td>9</td>
      <td>17/07/2024</td>
      <td>Mysuru</td>
      <td>No</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Lokesh</td>
      <td>Male</td>
      <td>21</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>75</td>
      <td>8</td>
      <td>2024/07/19</td>
      <td>Dharwad</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Geetha</td>
      <td>female</td>
      <td>23</td>
      <td>87</td>
      <td>3.5</td>
      <td>84</td>
      <td>9</td>
      <td>19-07-2024</td>
      <td>Hubli</td>
      <td>N</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Karthik</td>
      <td>M</td>
      <td>24</td>
      <td>70</td>
      <td>2.0</td>
      <td>62</td>
      <td>6</td>
      <td>2024-07-21</td>
      <td>Belagavi</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Divakar</td>
      <td>Male</td>
      <td>22</td>
      <td>75</td>
      <td>2.5</td>
      <td>66</td>
      <td>NaN</td>
      <td>21/07/2024</td>
      <td>Tumakuru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Nisha</td>
      <td>F</td>
      <td>21</td>
      <td>93</td>
      <td>4.5</td>
      <td>90</td>
      <td>10</td>
      <td>2024/07/23</td>
      <td>Shivamogga</td>
      <td>No</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Suraj</td>
      <td>male</td>
      <td>20</td>
      <td>68</td>
      <td>1.5</td>
      <td>57</td>
      <td>5</td>
      <td>23-07-2024</td>
      <td>Mangaluru</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Keerthi</td>
      <td>Female</td>
      <td>22</td>
      <td>86</td>
      <td>3.5</td>
      <td>83</td>
      <td>9</td>
      <td>2024.07.25</td>
      <td>Udupi</td>
      <td>No</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Manjunath</td>
      <td>M</td>
      <td>25</td>
      <td>62</td>
      <td>1.0</td>
      <td>49</td>
      <td>4</td>
      <td>25/07/2024</td>
      <td>Bengaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Pavithra</td>
      <td>female</td>
      <td>20</td>
      <td>89</td>
      <td>4.0</td>
      <td>86</td>
      <td>9</td>
      <td>2024/07/27</td>
      <td>Mysore</td>
      <td>No</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Rakesh</td>
      <td>Female</td>
      <td>23</td>
      <td>81</td>
      <td>3.0</td>
      <td>74</td>
      <td>8</td>
      <td>27-07-2024</td>
      <td>Dharwad</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Sahana</td>
      <td>F</td>
      <td>21</td>
      <td>95</td>
      <td>5.0</td>
      <td>93</td>
      <td>10</td>
      <td>2024-07-29</td>
      <td>Hubballi</td>
      <td>No</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Tejas</td>
      <td>M</td>
      <td>22</td>
      <td>76</td>
      <td>2.5</td>
      <td>65</td>
      <td>7</td>
      <td>29/07/2024</td>
      <td>Belgaum</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Arjun</td>
      <td>M</td>
      <td>22</td>
      <td>79%</td>
      <td>2.5</td>
      <td>69</td>
      <td>7</td>
      <td>2024-06-22</td>
      <td>Bengaluru</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Swathi</td>
      <td>F</td>
      <td>20</td>
      <td>91</td>
      <td>4.5</td>
      <td>94</td>
      <td>10</td>
      <td>07/07/2024</td>
      <td>Belgaum</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Study_Hours</th>
      <th>Previous_Marks</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>49.000000</td>
      <td>50.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.122449</td>
      <td>75.520000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.256361</td>
      <td>14.195774</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.500000</td>
      <td>45.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2.500000</td>
      <td>65.250000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.000000</td>
      <td>75.500000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>4.000000</td>
      <td>88.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>5.500000</td>
      <td>96.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.info()
```

    <class 'pandas.DataFrame'>
    RangeIndex: 50 entries, 0 to 49
    Data columns (total 10 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   Name                   50 non-null     str    
     1   Gender                 50 non-null     str    
     2   Age                    49 non-null     str    
     3   Attendance (%)         48 non-null     str    
     4   Study_Hours            49 non-null     float64
     5   Previous_Marks         50 non-null     int64  
     6   Assignments_Completed  49 non-null     str    
     7   Join_Date              50 non-null     str    
     8   City                   50 non-null     str    
     9   Extra_Classes          50 non-null     str    
    dtypes: float64(1), int64(1), str(8)
    memory usage: 4.0 KB
    


```python
df.isnull().sum()
```




    Name                     0
    Gender                   0
    Age                      1
    Attendance (%)           2
    Study_Hours              1
    Previous_Marks           0
    Assignments_Completed    1
    Join_Date                0
    City                     0
    Extra_Classes            0
    dtype: int64




```python
df.columns
```




    Index(['Name', 'Gender', 'Age', 'Attendance (%)', 'Study_Hours',
           'Previous_Marks', 'Assignments_Completed', 'Join_Date', 'City',
           'Extra_Classes'],
          dtype='str')




```python
df.duplicated().sum()
```




    np.int64(2)




```python
df.drop_duplicates()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Attendance (%)</th>
      <th>Study_Hours</th>
      <th>Previous_Marks</th>
      <th>Assignments_Completed</th>
      <th>Join_Date</th>
      <th>City</th>
      <th>Extra_Classes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rahul</td>
      <td>M</td>
      <td>21</td>
      <td>92</td>
      <td>3.5</td>
      <td>78</td>
      <td>9</td>
      <td>2024-06-10</td>
      <td>Bengaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Priya</td>
      <td>Female</td>
      <td>22</td>
      <td>88%</td>
      <td>4.0</td>
      <td>85</td>
      <td>10</td>
      <td>10/06/2024</td>
      <td>Bangalore</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Amit</td>
      <td>MALE</td>
      <td>20</td>
      <td>76</td>
      <td>2.5</td>
      <td>67</td>
      <td>8</td>
      <td>2024/06/12</td>
      <td>Mysuru</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sneha</td>
      <td>F</td>
      <td>23</td>
      <td>95</td>
      <td>5.0</td>
      <td>91</td>
      <td>10</td>
      <td>12-06-2024</td>
      <td>Mysore</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kiran</td>
      <td>Male</td>
      <td>21</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>72</td>
      <td>7</td>
      <td>2024-06-15</td>
      <td>Hubballi</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Anjali</td>
      <td>female</td>
      <td>20</td>
      <td>89</td>
      <td>4.5</td>
      <td>88</td>
      <td>9</td>
      <td>15/06/2024</td>
      <td>Hubli</td>
      <td>No</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Vikram</td>
      <td>M</td>
      <td>24</td>
      <td>72</td>
      <td>2.0</td>
      <td>61</td>
      <td>6</td>
      <td>2024-06-18</td>
      <td>Dharwad</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Neha</td>
      <td>Female</td>
      <td>22</td>
      <td>91</td>
      <td>NaN</td>
      <td>93</td>
      <td>10</td>
      <td>18/06/2024</td>
      <td>Dharwad</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Rohan</td>
      <td>Male</td>
      <td>19</td>
      <td>68</td>
      <td>1.5</td>
      <td>55</td>
      <td>5</td>
      <td>2024/06/20</td>
      <td>Belagavi</td>
      <td>No</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Divya</td>
      <td>FEMALE</td>
      <td>21</td>
      <td>84</td>
      <td>3.0</td>
      <td>76</td>
      <td>8</td>
      <td>20-06-2024</td>
      <td>Belgaum</td>
      <td>N</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Arjun</td>
      <td>M</td>
      <td>22</td>
      <td>79%</td>
      <td>2.5</td>
      <td>69</td>
      <td>7</td>
      <td>2024-06-22</td>
      <td>Bengaluru</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Pooja</td>
      <td>Female</td>
      <td>23</td>
      <td>96</td>
      <td>5.5</td>
      <td>95</td>
      <td>10</td>
      <td>22/06/2024</td>
      <td>Mysuru</td>
      <td>No</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Manoj</td>
      <td>Male</td>
      <td>twenty-five</td>
      <td>64</td>
      <td>1.0</td>
      <td>48</td>
      <td>4</td>
      <td>2024-06-25</td>
      <td>Tumakuru</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Asha</td>
      <td>F</td>
      <td>20</td>
      <td>87</td>
      <td>3.5</td>
      <td>82</td>
      <td>9</td>
      <td>25/06/2024</td>
      <td>Tumkur</td>
      <td>N</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Suresh</td>
      <td>male</td>
      <td>22</td>
      <td>73.5</td>
      <td>2.0</td>
      <td>64</td>
      <td>6</td>
      <td>2024/06/27</td>
      <td>Shivamogga</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Meena</td>
      <td>Female</td>
      <td>21</td>
      <td>90</td>
      <td>4.0</td>
      <td>89</td>
      <td>10</td>
      <td>27-06-2024</td>
      <td>Shimoga</td>
      <td>No</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Naveen</td>
      <td>M</td>
      <td>23</td>
      <td>78</td>
      <td>2.5</td>
      <td>71</td>
      <td>7</td>
      <td>2024-06-29</td>
      <td>Mangaluru</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kavya</td>
      <td>female</td>
      <td>20</td>
      <td>93</td>
      <td>4.5</td>
      <td>90</td>
      <td>10</td>
      <td>29/06/2024</td>
      <td>Mangalore</td>
      <td>No</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Prakash</td>
      <td>Male</td>
      <td>24</td>
      <td>69%</td>
      <td>1.5</td>
      <td>58</td>
      <td>5</td>
      <td>2024-07-01</td>
      <td>Udupi</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Lakshmi</td>
      <td>F</td>
      <td>22</td>
      <td>86%</td>
      <td>3.0</td>
      <td>81</td>
      <td>8</td>
      <td>01/07/2024</td>
      <td>Udupi</td>
      <td>N</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Harish</td>
      <td>M</td>
      <td>21</td>
      <td>82</td>
      <td>3.0</td>
      <td>74</td>
      <td>8</td>
      <td>2024/07/03</td>
      <td>Bengaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Shalini</td>
      <td>Female</td>
      <td>23</td>
      <td>94</td>
      <td>5.0</td>
      <td>92</td>
      <td>10</td>
      <td>03-07-2024</td>
      <td>Mysuru</td>
      <td>No</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Ganesh</td>
      <td>male</td>
      <td>26</td>
      <td>105</td>
      <td>0.5</td>
      <td>45</td>
      <td>3</td>
      <td>2024-07-05</td>
      <td>Dharwad</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Deepa</td>
      <td>Female</td>
      <td>21</td>
      <td>89</td>
      <td>4.0</td>
      <td>87</td>
      <td>9</td>
      <td>05/07/2024</td>
      <td>Hubballi</td>
      <td>No</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Ravi</td>
      <td>M</td>
      <td>22</td>
      <td>77</td>
      <td>2.5</td>
      <td>68</td>
      <td>seven</td>
      <td>2024-07-07</td>
      <td>Belagavi</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Swathi</td>
      <td>F</td>
      <td>20</td>
      <td>91</td>
      <td>4.5</td>
      <td>94</td>
      <td>10</td>
      <td>07/07/2024</td>
      <td>Belgaum</td>
      <td>No</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Mahesh</td>
      <td>Male</td>
      <td>23</td>
      <td>74</td>
      <td>2.0</td>
      <td>63</td>
      <td>6</td>
      <td>2024/07/09</td>
      <td>Tumakuru</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Nandini</td>
      <td>Female</td>
      <td>22</td>
      <td>88</td>
      <td>3.5</td>
      <td>85</td>
      <td>9</td>
      <td>09-07-2024</td>
      <td>Tumkur</td>
      <td>N</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Ajay</td>
      <td>M</td>
      <td>21</td>
      <td>80</td>
      <td>3.0</td>
      <td>73</td>
      <td>8</td>
      <td>2024-07-11</td>
      <td>Shivamogga</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Rekha</td>
      <td>female</td>
      <td>24</td>
      <td>97</td>
      <td>5.5</td>
      <td>96</td>
      <td>10</td>
      <td>11/07/2024</td>
      <td>Shimoga</td>
      <td>No</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Vijay</td>
      <td>Male</td>
      <td>25</td>
      <td>66</td>
      <td>1.0</td>
      <td>51</td>
      <td>4</td>
      <td>2024-07-13</td>
      <td>Mangaluru</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Sowmya</td>
      <td>F</td>
      <td>21</td>
      <td>85</td>
      <td>3.5</td>
      <td>79</td>
      <td>8</td>
      <td>13/07/2024</td>
      <td>Mangalore</td>
      <td>No</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Ramesh</td>
      <td>male</td>
      <td>22</td>
      <td>71</td>
      <td>2.0</td>
      <td>60</td>
      <td>6</td>
      <td>2024/07/15</td>
      <td>Udupi</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Bhavana</td>
      <td>Female</td>
      <td>NaN</td>
      <td>92</td>
      <td>4.0</td>
      <td>91</td>
      <td>10</td>
      <td>15-07-2024</td>
      <td>Udupi</td>
      <td>No</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Sanjay</td>
      <td>M</td>
      <td>23</td>
      <td>79</td>
      <td>2.5</td>
      <td>70</td>
      <td>7</td>
      <td>2024-07-17</td>
      <td>Bengaluru</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Aishwarya</td>
      <td>FEMALE</td>
      <td>22</td>
      <td>90</td>
      <td>4.5</td>
      <td>88</td>
      <td>9</td>
      <td>17/07/2024</td>
      <td>Mysuru</td>
      <td>No</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Lokesh</td>
      <td>Male</td>
      <td>21</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>75</td>
      <td>8</td>
      <td>2024/07/19</td>
      <td>Dharwad</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Geetha</td>
      <td>female</td>
      <td>23</td>
      <td>87</td>
      <td>3.5</td>
      <td>84</td>
      <td>9</td>
      <td>19-07-2024</td>
      <td>Hubli</td>
      <td>N</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Karthik</td>
      <td>M</td>
      <td>24</td>
      <td>70</td>
      <td>2.0</td>
      <td>62</td>
      <td>6</td>
      <td>2024-07-21</td>
      <td>Belagavi</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Divakar</td>
      <td>Male</td>
      <td>22</td>
      <td>75</td>
      <td>2.5</td>
      <td>66</td>
      <td>NaN</td>
      <td>21/07/2024</td>
      <td>Tumakuru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Nisha</td>
      <td>F</td>
      <td>21</td>
      <td>93</td>
      <td>4.5</td>
      <td>90</td>
      <td>10</td>
      <td>2024/07/23</td>
      <td>Shivamogga</td>
      <td>No</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Suraj</td>
      <td>male</td>
      <td>20</td>
      <td>68</td>
      <td>1.5</td>
      <td>57</td>
      <td>5</td>
      <td>23-07-2024</td>
      <td>Mangaluru</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Keerthi</td>
      <td>Female</td>
      <td>22</td>
      <td>86</td>
      <td>3.5</td>
      <td>83</td>
      <td>9</td>
      <td>2024.07.25</td>
      <td>Udupi</td>
      <td>No</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Manjunath</td>
      <td>M</td>
      <td>25</td>
      <td>62</td>
      <td>1.0</td>
      <td>49</td>
      <td>4</td>
      <td>25/07/2024</td>
      <td>Bengaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Pavithra</td>
      <td>female</td>
      <td>20</td>
      <td>89</td>
      <td>4.0</td>
      <td>86</td>
      <td>9</td>
      <td>2024/07/27</td>
      <td>Mysore</td>
      <td>No</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Rakesh</td>
      <td>Female</td>
      <td>23</td>
      <td>81</td>
      <td>3.0</td>
      <td>74</td>
      <td>8</td>
      <td>27-07-2024</td>
      <td>Dharwad</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Sahana</td>
      <td>F</td>
      <td>21</td>
      <td>95</td>
      <td>5.0</td>
      <td>93</td>
      <td>10</td>
      <td>2024-07-29</td>
      <td>Hubballi</td>
      <td>No</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Tejas</td>
      <td>M</td>
      <td>22</td>
      <td>76</td>
      <td>2.5</td>
      <td>65</td>
      <td>7</td>
      <td>29/07/2024</td>
      <td>Belgaum</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.shape
```




    (50, 10)




```python
text_col = ['Name', 'Gender', 'City','Extra_Classes']

for col in text_col:
    df[col]=df[col].str.strip()
```


```python
df['Gender']=df['Gender'].str.lower()
```


```python
df["Gender"]=df["Gender"].replace({"m" : "Male",
                                    "male":"Male",
                                    "female":"Female",
                                    "f":"Female"})
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Attendance (%)</th>
      <th>Study_Hours</th>
      <th>Previous_Marks</th>
      <th>Assignments_Completed</th>
      <th>Join_Date</th>
      <th>City</th>
      <th>Extra_Classes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rahul</td>
      <td>Male</td>
      <td>21</td>
      <td>92</td>
      <td>3.5</td>
      <td>78</td>
      <td>9</td>
      <td>2024-06-10</td>
      <td>Bengaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Priya</td>
      <td>Female</td>
      <td>22</td>
      <td>88%</td>
      <td>4.0</td>
      <td>85</td>
      <td>10</td>
      <td>10/06/2024</td>
      <td>Bangalore</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Amit</td>
      <td>Male</td>
      <td>20</td>
      <td>76</td>
      <td>2.5</td>
      <td>67</td>
      <td>8</td>
      <td>2024/06/12</td>
      <td>Mysuru</td>
      <td>Y</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sneha</td>
      <td>Female</td>
      <td>23</td>
      <td>95</td>
      <td>5.0</td>
      <td>91</td>
      <td>10</td>
      <td>12-06-2024</td>
      <td>Mysore</td>
      <td>N</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kiran</td>
      <td>Male</td>
      <td>21</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>72</td>
      <td>7</td>
      <td>2024-06-15</td>
      <td>Hubballi</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Extra_Classes']=df['Extra_Classes'].str.lower()
```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Attendance (%)</th>
      <th>Study_Hours</th>
      <th>Previous_Marks</th>
      <th>Assignments_Completed</th>
      <th>Join_Date</th>
      <th>City</th>
      <th>Extra_Classes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rahul</td>
      <td>Male</td>
      <td>21</td>
      <td>92</td>
      <td>3.5</td>
      <td>78</td>
      <td>9</td>
      <td>2024-06-10</td>
      <td>Bengaluru</td>
      <td>yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Priya</td>
      <td>Female</td>
      <td>22</td>
      <td>88%</td>
      <td>4.0</td>
      <td>85</td>
      <td>10</td>
      <td>10/06/2024</td>
      <td>Bangalore</td>
      <td>no</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Amit</td>
      <td>Male</td>
      <td>20</td>
      <td>76</td>
      <td>2.5</td>
      <td>67</td>
      <td>8</td>
      <td>2024/06/12</td>
      <td>Mysuru</td>
      <td>y</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sneha</td>
      <td>Female</td>
      <td>23</td>
      <td>95</td>
      <td>5.0</td>
      <td>91</td>
      <td>10</td>
      <td>12-06-2024</td>
      <td>Mysore</td>
      <td>n</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kiran</td>
      <td>Male</td>
      <td>21</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>72</td>
      <td>7</td>
      <td>2024-06-15</td>
      <td>Hubballi</td>
      <td>yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
Extra_Classes = {
    'yes': 'Yes',
    'y': 'Yes',
    'no': 'No',
    'n': 'No'
}

# Apply the map
df['Extra_Classes'] = df['Extra_Classes'].map(Extra_Classes)

```


```python
df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Attendance (%)</th>
      <th>Study_Hours</th>
      <th>Previous_Marks</th>
      <th>Assignments_Completed</th>
      <th>Join_Date</th>
      <th>City</th>
      <th>Extra_Classes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rahul</td>
      <td>Male</td>
      <td>21</td>
      <td>92</td>
      <td>3.5</td>
      <td>78</td>
      <td>9</td>
      <td>2024-06-10</td>
      <td>Bengaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Priya</td>
      <td>Female</td>
      <td>22</td>
      <td>88%</td>
      <td>4.0</td>
      <td>85</td>
      <td>10</td>
      <td>10/06/2024</td>
      <td>Bangalore</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Amit</td>
      <td>Male</td>
      <td>20</td>
      <td>76</td>
      <td>2.5</td>
      <td>67</td>
      <td>8</td>
      <td>2024/06/12</td>
      <td>Mysuru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sneha</td>
      <td>Female</td>
      <td>23</td>
      <td>95</td>
      <td>5.0</td>
      <td>91</td>
      <td>10</td>
      <td>12-06-2024</td>
      <td>Mysore</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kiran</td>
      <td>Male</td>
      <td>21</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>72</td>
      <td>7</td>
      <td>2024-06-15</td>
      <td>Hubballi</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Age'] =pd.to_numeric(df['Age'], errors='coerce')
```


```python
df['Attendance (%)'] =pd.to_numeric(df['Attendance (%)'], errors='coerce')
```


```python
df['Study_Hours'] =pd.to_numeric(df['Study_Hours'], errors='coerce')
```


```python
df['Previous_Marks'] =pd.to_numeric(df['Previous_Marks'], errors='coerce')
```


```python
df['Assignments_Completed'] =pd.to_numeric(df['Assignments_Completed'], errors='coerce')
```


```python
df.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Attendance (%)</th>
      <th>Study_Hours</th>
      <th>Previous_Marks</th>
      <th>Assignments_Completed</th>
      <th>Join_Date</th>
      <th>City</th>
      <th>Extra_Classes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rahul</td>
      <td>Male</td>
      <td>21.0</td>
      <td>92.0</td>
      <td>3.5</td>
      <td>78</td>
      <td>9.0</td>
      <td>2024-06-10</td>
      <td>Bengaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Priya</td>
      <td>Female</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>85</td>
      <td>10.0</td>
      <td>10/06/2024</td>
      <td>Bangalore</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Amit</td>
      <td>Male</td>
      <td>20.0</td>
      <td>76.0</td>
      <td>2.5</td>
      <td>67</td>
      <td>8.0</td>
      <td>2024/06/12</td>
      <td>Mysuru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sneha</td>
      <td>Female</td>
      <td>23.0</td>
      <td>95.0</td>
      <td>5.0</td>
      <td>91</td>
      <td>10.0</td>
      <td>12-06-2024</td>
      <td>Mysore</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kiran</td>
      <td>Male</td>
      <td>21.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>72</td>
      <td>7.0</td>
      <td>2024-06-15</td>
      <td>Hubballi</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Anjali</td>
      <td>Female</td>
      <td>20.0</td>
      <td>89.0</td>
      <td>4.5</td>
      <td>88</td>
      <td>9.0</td>
      <td>15/06/2024</td>
      <td>Hubli</td>
      <td>No</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Vikram</td>
      <td>Male</td>
      <td>24.0</td>
      <td>72.0</td>
      <td>2.0</td>
      <td>61</td>
      <td>6.0</td>
      <td>2024-06-18</td>
      <td>Dharwad</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Neha</td>
      <td>Female</td>
      <td>22.0</td>
      <td>91.0</td>
      <td>NaN</td>
      <td>93</td>
      <td>10.0</td>
      <td>18/06/2024</td>
      <td>Dharwad</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Rohan</td>
      <td>Male</td>
      <td>19.0</td>
      <td>68.0</td>
      <td>1.5</td>
      <td>55</td>
      <td>5.0</td>
      <td>2024/06/20</td>
      <td>Belagavi</td>
      <td>No</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Divya</td>
      <td>Female</td>
      <td>21.0</td>
      <td>84.0</td>
      <td>3.0</td>
      <td>76</td>
      <td>8.0</td>
      <td>20-06-2024</td>
      <td>Belgaum</td>
      <td>No</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Arjun</td>
      <td>Male</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>2.5</td>
      <td>69</td>
      <td>7.0</td>
      <td>2024-06-22</td>
      <td>Bengaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Pooja</td>
      <td>Female</td>
      <td>23.0</td>
      <td>96.0</td>
      <td>5.5</td>
      <td>95</td>
      <td>10.0</td>
      <td>22/06/2024</td>
      <td>Mysuru</td>
      <td>No</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Manoj</td>
      <td>Male</td>
      <td>NaN</td>
      <td>64.0</td>
      <td>1.0</td>
      <td>48</td>
      <td>4.0</td>
      <td>2024-06-25</td>
      <td>Tumakuru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Asha</td>
      <td>Female</td>
      <td>20.0</td>
      <td>87.0</td>
      <td>3.5</td>
      <td>82</td>
      <td>9.0</td>
      <td>25/06/2024</td>
      <td>Tumkur</td>
      <td>No</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Suresh</td>
      <td>Male</td>
      <td>22.0</td>
      <td>73.5</td>
      <td>2.0</td>
      <td>64</td>
      <td>6.0</td>
      <td>2024/06/27</td>
      <td>Shivamogga</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Meena</td>
      <td>Female</td>
      <td>21.0</td>
      <td>90.0</td>
      <td>4.0</td>
      <td>89</td>
      <td>10.0</td>
      <td>27-06-2024</td>
      <td>Shimoga</td>
      <td>No</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Naveen</td>
      <td>Male</td>
      <td>23.0</td>
      <td>78.0</td>
      <td>2.5</td>
      <td>71</td>
      <td>7.0</td>
      <td>2024-06-29</td>
      <td>Mangaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kavya</td>
      <td>Female</td>
      <td>20.0</td>
      <td>93.0</td>
      <td>4.5</td>
      <td>90</td>
      <td>10.0</td>
      <td>29/06/2024</td>
      <td>Mangalore</td>
      <td>No</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Prakash</td>
      <td>Male</td>
      <td>24.0</td>
      <td>NaN</td>
      <td>1.5</td>
      <td>58</td>
      <td>5.0</td>
      <td>2024-07-01</td>
      <td>Udupi</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Lakshmi</td>
      <td>Female</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>81</td>
      <td>8.0</td>
      <td>01/07/2024</td>
      <td>Udupi</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.isnull().sum()
```




    Name                     0
    Gender                   0
    Age                      2
    Attendance (%)           7
    Study_Hours              1
    Previous_Marks           0
    Assignments_Completed    2
    Join_Date                0
    City                     0
    Extra_Classes            0
    dtype: int64




```python
df.info()
```

    <class 'pandas.DataFrame'>
    RangeIndex: 50 entries, 0 to 49
    Data columns (total 10 columns):
     #   Column                 Non-Null Count  Dtype  
    ---  ------                 --------------  -----  
     0   Name                   50 non-null     str    
     1   Gender                 50 non-null     str    
     2   Age                    48 non-null     float64
     3   Attendance (%)         43 non-null     float64
     4   Study_Hours            49 non-null     float64
     5   Previous_Marks         50 non-null     int64  
     6   Assignments_Completed  48 non-null     float64
     7   Join_Date              50 non-null     str    
     8   City                   50 non-null     str    
     9   Extra_Classes          50 non-null     str    
    dtypes: float64(4), int64(1), str(5)
    memory usage: 4.0 KB
    


```python
df.head(20)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Attendance (%)</th>
      <th>Study_Hours</th>
      <th>Previous_Marks</th>
      <th>Assignments_Completed</th>
      <th>Join_Date</th>
      <th>City</th>
      <th>Extra_Classes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rahul</td>
      <td>Male</td>
      <td>21.0</td>
      <td>92.0</td>
      <td>3.5</td>
      <td>78</td>
      <td>9.0</td>
      <td>2024-06-10</td>
      <td>Bengaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Priya</td>
      <td>Female</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>85</td>
      <td>10.0</td>
      <td>10/06/2024</td>
      <td>Bangalore</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Amit</td>
      <td>Male</td>
      <td>20.0</td>
      <td>76.0</td>
      <td>2.5</td>
      <td>67</td>
      <td>8.0</td>
      <td>2024/06/12</td>
      <td>Mysuru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sneha</td>
      <td>Female</td>
      <td>23.0</td>
      <td>95.0</td>
      <td>5.0</td>
      <td>91</td>
      <td>10.0</td>
      <td>12-06-2024</td>
      <td>Mysore</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kiran</td>
      <td>Male</td>
      <td>21.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>72</td>
      <td>7.0</td>
      <td>2024-06-15</td>
      <td>Hubballi</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Anjali</td>
      <td>Female</td>
      <td>20.0</td>
      <td>89.0</td>
      <td>4.5</td>
      <td>88</td>
      <td>9.0</td>
      <td>15/06/2024</td>
      <td>Hubli</td>
      <td>No</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Vikram</td>
      <td>Male</td>
      <td>24.0</td>
      <td>72.0</td>
      <td>2.0</td>
      <td>61</td>
      <td>6.0</td>
      <td>2024-06-18</td>
      <td>Dharwad</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Neha</td>
      <td>Female</td>
      <td>22.0</td>
      <td>91.0</td>
      <td>NaN</td>
      <td>93</td>
      <td>10.0</td>
      <td>18/06/2024</td>
      <td>Dharwad</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Rohan</td>
      <td>Male</td>
      <td>19.0</td>
      <td>68.0</td>
      <td>1.5</td>
      <td>55</td>
      <td>5.0</td>
      <td>2024/06/20</td>
      <td>Belagavi</td>
      <td>No</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Divya</td>
      <td>Female</td>
      <td>21.0</td>
      <td>84.0</td>
      <td>3.0</td>
      <td>76</td>
      <td>8.0</td>
      <td>20-06-2024</td>
      <td>Belgaum</td>
      <td>No</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Arjun</td>
      <td>Male</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>2.5</td>
      <td>69</td>
      <td>7.0</td>
      <td>2024-06-22</td>
      <td>Bengaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Pooja</td>
      <td>Female</td>
      <td>23.0</td>
      <td>96.0</td>
      <td>5.5</td>
      <td>95</td>
      <td>10.0</td>
      <td>22/06/2024</td>
      <td>Mysuru</td>
      <td>No</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Manoj</td>
      <td>Male</td>
      <td>NaN</td>
      <td>64.0</td>
      <td>1.0</td>
      <td>48</td>
      <td>4.0</td>
      <td>2024-06-25</td>
      <td>Tumakuru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Asha</td>
      <td>Female</td>
      <td>20.0</td>
      <td>87.0</td>
      <td>3.5</td>
      <td>82</td>
      <td>9.0</td>
      <td>25/06/2024</td>
      <td>Tumkur</td>
      <td>No</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Suresh</td>
      <td>Male</td>
      <td>22.0</td>
      <td>73.5</td>
      <td>2.0</td>
      <td>64</td>
      <td>6.0</td>
      <td>2024/06/27</td>
      <td>Shivamogga</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Meena</td>
      <td>Female</td>
      <td>21.0</td>
      <td>90.0</td>
      <td>4.0</td>
      <td>89</td>
      <td>10.0</td>
      <td>27-06-2024</td>
      <td>Shimoga</td>
      <td>No</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Naveen</td>
      <td>Male</td>
      <td>23.0</td>
      <td>78.0</td>
      <td>2.5</td>
      <td>71</td>
      <td>7.0</td>
      <td>2024-06-29</td>
      <td>Mangaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kavya</td>
      <td>Female</td>
      <td>20.0</td>
      <td>93.0</td>
      <td>4.5</td>
      <td>90</td>
      <td>10.0</td>
      <td>29/06/2024</td>
      <td>Mangalore</td>
      <td>No</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Prakash</td>
      <td>Male</td>
      <td>24.0</td>
      <td>NaN</td>
      <td>1.5</td>
      <td>58</td>
      <td>5.0</td>
      <td>2024-07-01</td>
      <td>Udupi</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Lakshmi</td>
      <td>Female</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>81</td>
      <td>8.0</td>
      <td>01/07/2024</td>
      <td>Udupi</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["City"]=df["City"].replace({
    'Bengaluru' : 'Bengaluru',
    'Bangalore' : 'Bengaluru',
    'Mysore' : 'Mysore',
    'Mysuru' : 'Mysore',
    'Shivamogga' : 'Shivamoga',
    'Shimoga' : 'Shivamoga',
    'Tumakuru' : 'Tumkur',
    'Tumkur' : 'Tumkur',
    'Hubballi' : 'Hubali',
    'Hubali' : 'Hubali',
    'Hubli' : 'Hubali',
    'Mangaluru' : 'Mangaluru',
    'Mangalore' : 'Mangaluru',
})
```


```python
df[~df['Attendance (%)'].between(0,100)]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Attendance (%)</th>
      <th>Study_Hours</th>
      <th>Previous_Marks</th>
      <th>Assignments_Completed</th>
      <th>Join_Date</th>
      <th>City</th>
      <th>Extra_Classes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>Priya</td>
      <td>Female</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>85</td>
      <td>10.0</td>
      <td>10/06/2024</td>
      <td>Bengaluru</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kiran</td>
      <td>Male</td>
      <td>21.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>72</td>
      <td>7.0</td>
      <td>2024-06-15</td>
      <td>Hubali</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Arjun</td>
      <td>Male</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>2.5</td>
      <td>69</td>
      <td>7.0</td>
      <td>2024-06-22</td>
      <td>Bengaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Prakash</td>
      <td>Male</td>
      <td>24.0</td>
      <td>NaN</td>
      <td>1.5</td>
      <td>58</td>
      <td>5.0</td>
      <td>2024-07-01</td>
      <td>Udupi</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Lakshmi</td>
      <td>Female</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>81</td>
      <td>8.0</td>
      <td>01/07/2024</td>
      <td>Udupi</td>
      <td>No</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Ganesh</td>
      <td>Male</td>
      <td>26.0</td>
      <td>105.0</td>
      <td>0.5</td>
      <td>45</td>
      <td>3.0</td>
      <td>2024-07-05</td>
      <td>Dharwad</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Lokesh</td>
      <td>Male</td>
      <td>21.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>75</td>
      <td>8.0</td>
      <td>2024/07/19</td>
      <td>Dharwad</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Arjun</td>
      <td>Male</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>2.5</td>
      <td>69</td>
      <td>7.0</td>
      <td>2024-06-22</td>
      <td>Bengaluru</td>
      <td>Yes</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.tail(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Attendance (%)</th>
      <th>Study_Hours</th>
      <th>Previous_Marks</th>
      <th>Assignments_Completed</th>
      <th>Join_Date</th>
      <th>City</th>
      <th>Extra_Classes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>40</th>
      <td>Nisha</td>
      <td>Female</td>
      <td>21.0</td>
      <td>93.0</td>
      <td>4.5</td>
      <td>90</td>
      <td>10.0</td>
      <td>2024/07/23</td>
      <td>Shivamoga</td>
      <td>No</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Suraj</td>
      <td>Male</td>
      <td>20.0</td>
      <td>68.0</td>
      <td>1.5</td>
      <td>57</td>
      <td>5.0</td>
      <td>23-07-2024</td>
      <td>Mangaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Keerthi</td>
      <td>Female</td>
      <td>22.0</td>
      <td>86.0</td>
      <td>3.5</td>
      <td>83</td>
      <td>9.0</td>
      <td>2024.07.25</td>
      <td>Udupi</td>
      <td>No</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Manjunath</td>
      <td>Male</td>
      <td>25.0</td>
      <td>62.0</td>
      <td>1.0</td>
      <td>49</td>
      <td>4.0</td>
      <td>25/07/2024</td>
      <td>Bengaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Pavithra</td>
      <td>Female</td>
      <td>20.0</td>
      <td>89.0</td>
      <td>4.0</td>
      <td>86</td>
      <td>9.0</td>
      <td>2024/07/27</td>
      <td>Mysore</td>
      <td>No</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Rakesh</td>
      <td>Female</td>
      <td>23.0</td>
      <td>81.0</td>
      <td>3.0</td>
      <td>74</td>
      <td>8.0</td>
      <td>27-07-2024</td>
      <td>Dharwad</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Sahana</td>
      <td>Female</td>
      <td>21.0</td>
      <td>95.0</td>
      <td>5.0</td>
      <td>93</td>
      <td>10.0</td>
      <td>2024-07-29</td>
      <td>Hubali</td>
      <td>No</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Tejas</td>
      <td>Male</td>
      <td>22.0</td>
      <td>76.0</td>
      <td>2.5</td>
      <td>65</td>
      <td>7.0</td>
      <td>29/07/2024</td>
      <td>Belgaum</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Arjun</td>
      <td>Male</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>2.5</td>
      <td>69</td>
      <td>7.0</td>
      <td>2024-06-22</td>
      <td>Bengaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Swathi</td>
      <td>Female</td>
      <td>20.0</td>
      <td>91.0</td>
      <td>4.5</td>
      <td>94</td>
      <td>10.0</td>
      <td>07/07/2024</td>
      <td>Belgaum</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
df.head(50)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Attendance (%)</th>
      <th>Study_Hours</th>
      <th>Previous_Marks</th>
      <th>Assignments_Completed</th>
      <th>Join_Date</th>
      <th>City</th>
      <th>Extra_Classes</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rahul</td>
      <td>Male</td>
      <td>21.0</td>
      <td>92.0</td>
      <td>3.5</td>
      <td>78</td>
      <td>9.0</td>
      <td>2024-06-10</td>
      <td>Bengaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Priya</td>
      <td>Female</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>85</td>
      <td>10.0</td>
      <td>10/06/2024</td>
      <td>Bengaluru</td>
      <td>No</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Amit</td>
      <td>Male</td>
      <td>20.0</td>
      <td>76.0</td>
      <td>2.5</td>
      <td>67</td>
      <td>8.0</td>
      <td>2024/06/12</td>
      <td>Mysore</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sneha</td>
      <td>Female</td>
      <td>23.0</td>
      <td>95.0</td>
      <td>5.0</td>
      <td>91</td>
      <td>10.0</td>
      <td>12-06-2024</td>
      <td>Mysore</td>
      <td>No</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kiran</td>
      <td>Male</td>
      <td>21.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>72</td>
      <td>7.0</td>
      <td>2024-06-15</td>
      <td>Hubali</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Anjali</td>
      <td>Female</td>
      <td>20.0</td>
      <td>89.0</td>
      <td>4.5</td>
      <td>88</td>
      <td>9.0</td>
      <td>15/06/2024</td>
      <td>Hubali</td>
      <td>No</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Vikram</td>
      <td>Male</td>
      <td>24.0</td>
      <td>72.0</td>
      <td>2.0</td>
      <td>61</td>
      <td>6.0</td>
      <td>2024-06-18</td>
      <td>Dharwad</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Neha</td>
      <td>Female</td>
      <td>22.0</td>
      <td>91.0</td>
      <td>NaN</td>
      <td>93</td>
      <td>10.0</td>
      <td>18/06/2024</td>
      <td>Dharwad</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Rohan</td>
      <td>Male</td>
      <td>19.0</td>
      <td>68.0</td>
      <td>1.5</td>
      <td>55</td>
      <td>5.0</td>
      <td>2024/06/20</td>
      <td>Belagavi</td>
      <td>No</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Divya</td>
      <td>Female</td>
      <td>21.0</td>
      <td>84.0</td>
      <td>3.0</td>
      <td>76</td>
      <td>8.0</td>
      <td>20-06-2024</td>
      <td>Belgaum</td>
      <td>No</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Arjun</td>
      <td>Male</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>2.5</td>
      <td>69</td>
      <td>7.0</td>
      <td>2024-06-22</td>
      <td>Bengaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Pooja</td>
      <td>Female</td>
      <td>23.0</td>
      <td>96.0</td>
      <td>5.5</td>
      <td>95</td>
      <td>10.0</td>
      <td>22/06/2024</td>
      <td>Mysore</td>
      <td>No</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Manoj</td>
      <td>Male</td>
      <td>NaN</td>
      <td>64.0</td>
      <td>1.0</td>
      <td>48</td>
      <td>4.0</td>
      <td>2024-06-25</td>
      <td>Tumkur</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Asha</td>
      <td>Female</td>
      <td>20.0</td>
      <td>87.0</td>
      <td>3.5</td>
      <td>82</td>
      <td>9.0</td>
      <td>25/06/2024</td>
      <td>Tumkur</td>
      <td>No</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Suresh</td>
      <td>Male</td>
      <td>22.0</td>
      <td>73.5</td>
      <td>2.0</td>
      <td>64</td>
      <td>6.0</td>
      <td>2024/06/27</td>
      <td>Shivamoga</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Meena</td>
      <td>Female</td>
      <td>21.0</td>
      <td>90.0</td>
      <td>4.0</td>
      <td>89</td>
      <td>10.0</td>
      <td>27-06-2024</td>
      <td>Shivamoga</td>
      <td>No</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Naveen</td>
      <td>Male</td>
      <td>23.0</td>
      <td>78.0</td>
      <td>2.5</td>
      <td>71</td>
      <td>7.0</td>
      <td>2024-06-29</td>
      <td>Mangaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kavya</td>
      <td>Female</td>
      <td>20.0</td>
      <td>93.0</td>
      <td>4.5</td>
      <td>90</td>
      <td>10.0</td>
      <td>29/06/2024</td>
      <td>Mangaluru</td>
      <td>No</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Prakash</td>
      <td>Male</td>
      <td>24.0</td>
      <td>NaN</td>
      <td>1.5</td>
      <td>58</td>
      <td>5.0</td>
      <td>2024-07-01</td>
      <td>Udupi</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Lakshmi</td>
      <td>Female</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>81</td>
      <td>8.0</td>
      <td>01/07/2024</td>
      <td>Udupi</td>
      <td>No</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Harish</td>
      <td>Male</td>
      <td>21.0</td>
      <td>82.0</td>
      <td>3.0</td>
      <td>74</td>
      <td>8.0</td>
      <td>2024/07/03</td>
      <td>Bengaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Shalini</td>
      <td>Female</td>
      <td>23.0</td>
      <td>94.0</td>
      <td>5.0</td>
      <td>92</td>
      <td>10.0</td>
      <td>03-07-2024</td>
      <td>Mysore</td>
      <td>No</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Ganesh</td>
      <td>Male</td>
      <td>26.0</td>
      <td>105.0</td>
      <td>0.5</td>
      <td>45</td>
      <td>3.0</td>
      <td>2024-07-05</td>
      <td>Dharwad</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Deepa</td>
      <td>Female</td>
      <td>21.0</td>
      <td>89.0</td>
      <td>4.0</td>
      <td>87</td>
      <td>9.0</td>
      <td>05/07/2024</td>
      <td>Hubali</td>
      <td>No</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Ravi</td>
      <td>Male</td>
      <td>22.0</td>
      <td>77.0</td>
      <td>2.5</td>
      <td>68</td>
      <td>NaN</td>
      <td>2024-07-07</td>
      <td>Belagavi</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Swathi</td>
      <td>Female</td>
      <td>20.0</td>
      <td>91.0</td>
      <td>4.5</td>
      <td>94</td>
      <td>10.0</td>
      <td>07/07/2024</td>
      <td>Belgaum</td>
      <td>No</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Mahesh</td>
      <td>Male</td>
      <td>23.0</td>
      <td>74.0</td>
      <td>2.0</td>
      <td>63</td>
      <td>6.0</td>
      <td>2024/07/09</td>
      <td>Tumkur</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Nandini</td>
      <td>Female</td>
      <td>22.0</td>
      <td>88.0</td>
      <td>3.5</td>
      <td>85</td>
      <td>9.0</td>
      <td>09-07-2024</td>
      <td>Tumkur</td>
      <td>No</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Ajay</td>
      <td>Male</td>
      <td>21.0</td>
      <td>80.0</td>
      <td>3.0</td>
      <td>73</td>
      <td>8.0</td>
      <td>2024-07-11</td>
      <td>Shivamoga</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Rekha</td>
      <td>Female</td>
      <td>24.0</td>
      <td>97.0</td>
      <td>5.5</td>
      <td>96</td>
      <td>10.0</td>
      <td>11/07/2024</td>
      <td>Shivamoga</td>
      <td>No</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Vijay</td>
      <td>Male</td>
      <td>25.0</td>
      <td>66.0</td>
      <td>1.0</td>
      <td>51</td>
      <td>4.0</td>
      <td>2024-07-13</td>
      <td>Mangaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Sowmya</td>
      <td>Female</td>
      <td>21.0</td>
      <td>85.0</td>
      <td>3.5</td>
      <td>79</td>
      <td>8.0</td>
      <td>13/07/2024</td>
      <td>Mangaluru</td>
      <td>No</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Ramesh</td>
      <td>Male</td>
      <td>22.0</td>
      <td>71.0</td>
      <td>2.0</td>
      <td>60</td>
      <td>6.0</td>
      <td>2024/07/15</td>
      <td>Udupi</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Bhavana</td>
      <td>Female</td>
      <td>NaN</td>
      <td>92.0</td>
      <td>4.0</td>
      <td>91</td>
      <td>10.0</td>
      <td>15-07-2024</td>
      <td>Udupi</td>
      <td>No</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Sanjay</td>
      <td>Male</td>
      <td>23.0</td>
      <td>79.0</td>
      <td>2.5</td>
      <td>70</td>
      <td>7.0</td>
      <td>2024-07-17</td>
      <td>Bengaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Aishwarya</td>
      <td>Female</td>
      <td>22.0</td>
      <td>90.0</td>
      <td>4.5</td>
      <td>88</td>
      <td>9.0</td>
      <td>17/07/2024</td>
      <td>Mysore</td>
      <td>No</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Lokesh</td>
      <td>Male</td>
      <td>21.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>75</td>
      <td>8.0</td>
      <td>2024/07/19</td>
      <td>Dharwad</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Geetha</td>
      <td>Female</td>
      <td>23.0</td>
      <td>87.0</td>
      <td>3.5</td>
      <td>84</td>
      <td>9.0</td>
      <td>19-07-2024</td>
      <td>Hubali</td>
      <td>No</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Karthik</td>
      <td>Male</td>
      <td>24.0</td>
      <td>70.0</td>
      <td>2.0</td>
      <td>62</td>
      <td>6.0</td>
      <td>2024-07-21</td>
      <td>Belagavi</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Divakar</td>
      <td>Male</td>
      <td>22.0</td>
      <td>75.0</td>
      <td>2.5</td>
      <td>66</td>
      <td>NaN</td>
      <td>21/07/2024</td>
      <td>Tumkur</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Nisha</td>
      <td>Female</td>
      <td>21.0</td>
      <td>93.0</td>
      <td>4.5</td>
      <td>90</td>
      <td>10.0</td>
      <td>2024/07/23</td>
      <td>Shivamoga</td>
      <td>No</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Suraj</td>
      <td>Male</td>
      <td>20.0</td>
      <td>68.0</td>
      <td>1.5</td>
      <td>57</td>
      <td>5.0</td>
      <td>23-07-2024</td>
      <td>Mangaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Keerthi</td>
      <td>Female</td>
      <td>22.0</td>
      <td>86.0</td>
      <td>3.5</td>
      <td>83</td>
      <td>9.0</td>
      <td>2024.07.25</td>
      <td>Udupi</td>
      <td>No</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Manjunath</td>
      <td>Male</td>
      <td>25.0</td>
      <td>62.0</td>
      <td>1.0</td>
      <td>49</td>
      <td>4.0</td>
      <td>25/07/2024</td>
      <td>Bengaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Pavithra</td>
      <td>Female</td>
      <td>20.0</td>
      <td>89.0</td>
      <td>4.0</td>
      <td>86</td>
      <td>9.0</td>
      <td>2024/07/27</td>
      <td>Mysore</td>
      <td>No</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Rakesh</td>
      <td>Female</td>
      <td>23.0</td>
      <td>81.0</td>
      <td>3.0</td>
      <td>74</td>
      <td>8.0</td>
      <td>27-07-2024</td>
      <td>Dharwad</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Sahana</td>
      <td>Female</td>
      <td>21.0</td>
      <td>95.0</td>
      <td>5.0</td>
      <td>93</td>
      <td>10.0</td>
      <td>2024-07-29</td>
      <td>Hubali</td>
      <td>No</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Tejas</td>
      <td>Male</td>
      <td>22.0</td>
      <td>76.0</td>
      <td>2.5</td>
      <td>65</td>
      <td>7.0</td>
      <td>29/07/2024</td>
      <td>Belgaum</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Arjun</td>
      <td>Male</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>2.5</td>
      <td>69</td>
      <td>7.0</td>
      <td>2024-06-22</td>
      <td>Bengaluru</td>
      <td>Yes</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Swathi</td>
      <td>Female</td>
      <td>20.0</td>
      <td>91.0</td>
      <td>4.5</td>
      <td>94</td>
      <td>10.0</td>
      <td>07/07/2024</td>
      <td>Belgaum</td>
      <td>No</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Join_Date_1'] = df['Join_Date'].str.strip().str.replace('/', '-')
```


```python
df['Join_Date_2'] =pd.to_datetime(df['Join_Date_1'], errors='coerce')
```


```python
df.head(10)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Attendance (%)</th>
      <th>Study_Hours</th>
      <th>Previous_Marks</th>
      <th>Assignments_Completed</th>
      <th>Join_Date</th>
      <th>City</th>
      <th>Extra_Classes</th>
      <th>Join_Date_1</th>
      <th>Join_Date_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rahul</td>
      <td>Male</td>
      <td>21.0</td>
      <td>92.0</td>
      <td>3.5</td>
      <td>78</td>
      <td>9.0</td>
      <td>2024-06-10</td>
      <td>Bengaluru</td>
      <td>Yes</td>
      <td>2024-06-10</td>
      <td>2024-06-10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Priya</td>
      <td>Female</td>
      <td>22.0</td>
      <td>NaN</td>
      <td>4.0</td>
      <td>85</td>
      <td>10.0</td>
      <td>10/06/2024</td>
      <td>Bengaluru</td>
      <td>No</td>
      <td>10-06-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Amit</td>
      <td>Male</td>
      <td>20.0</td>
      <td>76.0</td>
      <td>2.5</td>
      <td>67</td>
      <td>8.0</td>
      <td>2024/06/12</td>
      <td>Mysore</td>
      <td>Yes</td>
      <td>2024-06-12</td>
      <td>2024-06-12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sneha</td>
      <td>Female</td>
      <td>23.0</td>
      <td>95.0</td>
      <td>5.0</td>
      <td>91</td>
      <td>10.0</td>
      <td>12-06-2024</td>
      <td>Mysore</td>
      <td>No</td>
      <td>12-06-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kiran</td>
      <td>Male</td>
      <td>21.0</td>
      <td>NaN</td>
      <td>3.0</td>
      <td>72</td>
      <td>7.0</td>
      <td>2024-06-15</td>
      <td>Hubali</td>
      <td>Yes</td>
      <td>2024-06-15</td>
      <td>2024-06-15</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Anjali</td>
      <td>Female</td>
      <td>20.0</td>
      <td>89.0</td>
      <td>4.5</td>
      <td>88</td>
      <td>9.0</td>
      <td>15/06/2024</td>
      <td>Hubali</td>
      <td>No</td>
      <td>15-06-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Vikram</td>
      <td>Male</td>
      <td>24.0</td>
      <td>72.0</td>
      <td>2.0</td>
      <td>61</td>
      <td>6.0</td>
      <td>2024-06-18</td>
      <td>Dharwad</td>
      <td>Yes</td>
      <td>2024-06-18</td>
      <td>2024-06-18</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Neha</td>
      <td>Female</td>
      <td>22.0</td>
      <td>91.0</td>
      <td>NaN</td>
      <td>93</td>
      <td>10.0</td>
      <td>18/06/2024</td>
      <td>Dharwad</td>
      <td>Yes</td>
      <td>18-06-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Rohan</td>
      <td>Male</td>
      <td>19.0</td>
      <td>68.0</td>
      <td>1.5</td>
      <td>55</td>
      <td>5.0</td>
      <td>2024/06/20</td>
      <td>Belagavi</td>
      <td>No</td>
      <td>2024-06-20</td>
      <td>2024-06-20</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Divya</td>
      <td>Female</td>
      <td>21.0</td>
      <td>84.0</td>
      <td>3.0</td>
      <td>76</td>
      <td>8.0</td>
      <td>20-06-2024</td>
      <td>Belgaum</td>
      <td>No</td>
      <td>20-06-2024</td>
      <td>NaT</td>
    </tr>
  </tbody>
</table>
</div>




```python
df['Age'] =df['Age'].fillna(df['Age'].median())
```


```python
df['Study_Hours'] =df['Study_Hours'].fillna(df['Study_Hours'].mean())
```


```python
df['Attendance (%)'] =df['Attendance (%)'].fillna(df['Attendance (%)'].mode()[0])
```


```python
df.to_csv("Students.csv")
```


```python
df.head(50)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Name</th>
      <th>Gender</th>
      <th>Age</th>
      <th>Attendance (%)</th>
      <th>Study_Hours</th>
      <th>Previous_Marks</th>
      <th>Assignments_Completed</th>
      <th>Join_Date</th>
      <th>City</th>
      <th>Extra_Classes</th>
      <th>Join_Date_1</th>
      <th>Join_Date_2</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Rahul</td>
      <td>Male</td>
      <td>21.0</td>
      <td>92.0</td>
      <td>3.500000</td>
      <td>78</td>
      <td>9.0</td>
      <td>2024-06-10</td>
      <td>Bengaluru</td>
      <td>Yes</td>
      <td>2024-06-10</td>
      <td>2024-06-10</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Priya</td>
      <td>Female</td>
      <td>22.0</td>
      <td>89.0</td>
      <td>4.000000</td>
      <td>85</td>
      <td>10.0</td>
      <td>10/06/2024</td>
      <td>Bengaluru</td>
      <td>No</td>
      <td>10-06-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Amit</td>
      <td>Male</td>
      <td>20.0</td>
      <td>76.0</td>
      <td>2.500000</td>
      <td>67</td>
      <td>8.0</td>
      <td>2024/06/12</td>
      <td>Mysore</td>
      <td>Yes</td>
      <td>2024-06-12</td>
      <td>2024-06-12</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Sneha</td>
      <td>Female</td>
      <td>23.0</td>
      <td>95.0</td>
      <td>5.000000</td>
      <td>91</td>
      <td>10.0</td>
      <td>12-06-2024</td>
      <td>Mysore</td>
      <td>No</td>
      <td>12-06-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kiran</td>
      <td>Male</td>
      <td>21.0</td>
      <td>89.0</td>
      <td>3.000000</td>
      <td>72</td>
      <td>7.0</td>
      <td>2024-06-15</td>
      <td>Hubali</td>
      <td>Yes</td>
      <td>2024-06-15</td>
      <td>2024-06-15</td>
    </tr>
    <tr>
      <th>5</th>
      <td>Anjali</td>
      <td>Female</td>
      <td>20.0</td>
      <td>89.0</td>
      <td>4.500000</td>
      <td>88</td>
      <td>9.0</td>
      <td>15/06/2024</td>
      <td>Hubali</td>
      <td>No</td>
      <td>15-06-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>6</th>
      <td>Vikram</td>
      <td>Male</td>
      <td>24.0</td>
      <td>72.0</td>
      <td>2.000000</td>
      <td>61</td>
      <td>6.0</td>
      <td>2024-06-18</td>
      <td>Dharwad</td>
      <td>Yes</td>
      <td>2024-06-18</td>
      <td>2024-06-18</td>
    </tr>
    <tr>
      <th>7</th>
      <td>Neha</td>
      <td>Female</td>
      <td>22.0</td>
      <td>91.0</td>
      <td>3.122449</td>
      <td>93</td>
      <td>10.0</td>
      <td>18/06/2024</td>
      <td>Dharwad</td>
      <td>Yes</td>
      <td>18-06-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>8</th>
      <td>Rohan</td>
      <td>Male</td>
      <td>19.0</td>
      <td>68.0</td>
      <td>1.500000</td>
      <td>55</td>
      <td>5.0</td>
      <td>2024/06/20</td>
      <td>Belagavi</td>
      <td>No</td>
      <td>2024-06-20</td>
      <td>2024-06-20</td>
    </tr>
    <tr>
      <th>9</th>
      <td>Divya</td>
      <td>Female</td>
      <td>21.0</td>
      <td>84.0</td>
      <td>3.000000</td>
      <td>76</td>
      <td>8.0</td>
      <td>20-06-2024</td>
      <td>Belgaum</td>
      <td>No</td>
      <td>20-06-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>10</th>
      <td>Arjun</td>
      <td>Male</td>
      <td>22.0</td>
      <td>89.0</td>
      <td>2.500000</td>
      <td>69</td>
      <td>7.0</td>
      <td>2024-06-22</td>
      <td>Bengaluru</td>
      <td>Yes</td>
      <td>2024-06-22</td>
      <td>2024-06-22</td>
    </tr>
    <tr>
      <th>11</th>
      <td>Pooja</td>
      <td>Female</td>
      <td>23.0</td>
      <td>96.0</td>
      <td>5.500000</td>
      <td>95</td>
      <td>10.0</td>
      <td>22/06/2024</td>
      <td>Mysore</td>
      <td>No</td>
      <td>22-06-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>12</th>
      <td>Manoj</td>
      <td>Male</td>
      <td>22.0</td>
      <td>64.0</td>
      <td>1.000000</td>
      <td>48</td>
      <td>4.0</td>
      <td>2024-06-25</td>
      <td>Tumkur</td>
      <td>Yes</td>
      <td>2024-06-25</td>
      <td>2024-06-25</td>
    </tr>
    <tr>
      <th>13</th>
      <td>Asha</td>
      <td>Female</td>
      <td>20.0</td>
      <td>87.0</td>
      <td>3.500000</td>
      <td>82</td>
      <td>9.0</td>
      <td>25/06/2024</td>
      <td>Tumkur</td>
      <td>No</td>
      <td>25-06-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>14</th>
      <td>Suresh</td>
      <td>Male</td>
      <td>22.0</td>
      <td>73.5</td>
      <td>2.000000</td>
      <td>64</td>
      <td>6.0</td>
      <td>2024/06/27</td>
      <td>Shivamoga</td>
      <td>Yes</td>
      <td>2024-06-27</td>
      <td>2024-06-27</td>
    </tr>
    <tr>
      <th>15</th>
      <td>Meena</td>
      <td>Female</td>
      <td>21.0</td>
      <td>90.0</td>
      <td>4.000000</td>
      <td>89</td>
      <td>10.0</td>
      <td>27-06-2024</td>
      <td>Shivamoga</td>
      <td>No</td>
      <td>27-06-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>16</th>
      <td>Naveen</td>
      <td>Male</td>
      <td>23.0</td>
      <td>78.0</td>
      <td>2.500000</td>
      <td>71</td>
      <td>7.0</td>
      <td>2024-06-29</td>
      <td>Mangaluru</td>
      <td>Yes</td>
      <td>2024-06-29</td>
      <td>2024-06-29</td>
    </tr>
    <tr>
      <th>17</th>
      <td>Kavya</td>
      <td>Female</td>
      <td>20.0</td>
      <td>93.0</td>
      <td>4.500000</td>
      <td>90</td>
      <td>10.0</td>
      <td>29/06/2024</td>
      <td>Mangaluru</td>
      <td>No</td>
      <td>29-06-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>18</th>
      <td>Prakash</td>
      <td>Male</td>
      <td>24.0</td>
      <td>89.0</td>
      <td>1.500000</td>
      <td>58</td>
      <td>5.0</td>
      <td>2024-07-01</td>
      <td>Udupi</td>
      <td>Yes</td>
      <td>2024-07-01</td>
      <td>2024-07-01</td>
    </tr>
    <tr>
      <th>19</th>
      <td>Lakshmi</td>
      <td>Female</td>
      <td>22.0</td>
      <td>89.0</td>
      <td>3.000000</td>
      <td>81</td>
      <td>8.0</td>
      <td>01/07/2024</td>
      <td>Udupi</td>
      <td>No</td>
      <td>01-07-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>20</th>
      <td>Harish</td>
      <td>Male</td>
      <td>21.0</td>
      <td>82.0</td>
      <td>3.000000</td>
      <td>74</td>
      <td>8.0</td>
      <td>2024/07/03</td>
      <td>Bengaluru</td>
      <td>Yes</td>
      <td>2024-07-03</td>
      <td>2024-07-03</td>
    </tr>
    <tr>
      <th>21</th>
      <td>Shalini</td>
      <td>Female</td>
      <td>23.0</td>
      <td>94.0</td>
      <td>5.000000</td>
      <td>92</td>
      <td>10.0</td>
      <td>03-07-2024</td>
      <td>Mysore</td>
      <td>No</td>
      <td>03-07-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>22</th>
      <td>Ganesh</td>
      <td>Male</td>
      <td>26.0</td>
      <td>105.0</td>
      <td>0.500000</td>
      <td>45</td>
      <td>3.0</td>
      <td>2024-07-05</td>
      <td>Dharwad</td>
      <td>Yes</td>
      <td>2024-07-05</td>
      <td>2024-07-05</td>
    </tr>
    <tr>
      <th>23</th>
      <td>Deepa</td>
      <td>Female</td>
      <td>21.0</td>
      <td>89.0</td>
      <td>4.000000</td>
      <td>87</td>
      <td>9.0</td>
      <td>05/07/2024</td>
      <td>Hubali</td>
      <td>No</td>
      <td>05-07-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>24</th>
      <td>Ravi</td>
      <td>Male</td>
      <td>22.0</td>
      <td>77.0</td>
      <td>2.500000</td>
      <td>68</td>
      <td>NaN</td>
      <td>2024-07-07</td>
      <td>Belagavi</td>
      <td>Yes</td>
      <td>2024-07-07</td>
      <td>2024-07-07</td>
    </tr>
    <tr>
      <th>25</th>
      <td>Swathi</td>
      <td>Female</td>
      <td>20.0</td>
      <td>91.0</td>
      <td>4.500000</td>
      <td>94</td>
      <td>10.0</td>
      <td>07/07/2024</td>
      <td>Belgaum</td>
      <td>No</td>
      <td>07-07-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>26</th>
      <td>Mahesh</td>
      <td>Male</td>
      <td>23.0</td>
      <td>74.0</td>
      <td>2.000000</td>
      <td>63</td>
      <td>6.0</td>
      <td>2024/07/09</td>
      <td>Tumkur</td>
      <td>Yes</td>
      <td>2024-07-09</td>
      <td>2024-07-09</td>
    </tr>
    <tr>
      <th>27</th>
      <td>Nandini</td>
      <td>Female</td>
      <td>22.0</td>
      <td>88.0</td>
      <td>3.500000</td>
      <td>85</td>
      <td>9.0</td>
      <td>09-07-2024</td>
      <td>Tumkur</td>
      <td>No</td>
      <td>09-07-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>28</th>
      <td>Ajay</td>
      <td>Male</td>
      <td>21.0</td>
      <td>80.0</td>
      <td>3.000000</td>
      <td>73</td>
      <td>8.0</td>
      <td>2024-07-11</td>
      <td>Shivamoga</td>
      <td>Yes</td>
      <td>2024-07-11</td>
      <td>2024-07-11</td>
    </tr>
    <tr>
      <th>29</th>
      <td>Rekha</td>
      <td>Female</td>
      <td>24.0</td>
      <td>97.0</td>
      <td>5.500000</td>
      <td>96</td>
      <td>10.0</td>
      <td>11/07/2024</td>
      <td>Shivamoga</td>
      <td>No</td>
      <td>11-07-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>30</th>
      <td>Vijay</td>
      <td>Male</td>
      <td>25.0</td>
      <td>66.0</td>
      <td>1.000000</td>
      <td>51</td>
      <td>4.0</td>
      <td>2024-07-13</td>
      <td>Mangaluru</td>
      <td>Yes</td>
      <td>2024-07-13</td>
      <td>2024-07-13</td>
    </tr>
    <tr>
      <th>31</th>
      <td>Sowmya</td>
      <td>Female</td>
      <td>21.0</td>
      <td>85.0</td>
      <td>3.500000</td>
      <td>79</td>
      <td>8.0</td>
      <td>13/07/2024</td>
      <td>Mangaluru</td>
      <td>No</td>
      <td>13-07-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>32</th>
      <td>Ramesh</td>
      <td>Male</td>
      <td>22.0</td>
      <td>71.0</td>
      <td>2.000000</td>
      <td>60</td>
      <td>6.0</td>
      <td>2024/07/15</td>
      <td>Udupi</td>
      <td>Yes</td>
      <td>2024-07-15</td>
      <td>2024-07-15</td>
    </tr>
    <tr>
      <th>33</th>
      <td>Bhavana</td>
      <td>Female</td>
      <td>22.0</td>
      <td>92.0</td>
      <td>4.000000</td>
      <td>91</td>
      <td>10.0</td>
      <td>15-07-2024</td>
      <td>Udupi</td>
      <td>No</td>
      <td>15-07-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>34</th>
      <td>Sanjay</td>
      <td>Male</td>
      <td>23.0</td>
      <td>79.0</td>
      <td>2.500000</td>
      <td>70</td>
      <td>7.0</td>
      <td>2024-07-17</td>
      <td>Bengaluru</td>
      <td>Yes</td>
      <td>2024-07-17</td>
      <td>2024-07-17</td>
    </tr>
    <tr>
      <th>35</th>
      <td>Aishwarya</td>
      <td>Female</td>
      <td>22.0</td>
      <td>90.0</td>
      <td>4.500000</td>
      <td>88</td>
      <td>9.0</td>
      <td>17/07/2024</td>
      <td>Mysore</td>
      <td>No</td>
      <td>17-07-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>36</th>
      <td>Lokesh</td>
      <td>Male</td>
      <td>21.0</td>
      <td>89.0</td>
      <td>3.000000</td>
      <td>75</td>
      <td>8.0</td>
      <td>2024/07/19</td>
      <td>Dharwad</td>
      <td>Yes</td>
      <td>2024-07-19</td>
      <td>2024-07-19</td>
    </tr>
    <tr>
      <th>37</th>
      <td>Geetha</td>
      <td>Female</td>
      <td>23.0</td>
      <td>87.0</td>
      <td>3.500000</td>
      <td>84</td>
      <td>9.0</td>
      <td>19-07-2024</td>
      <td>Hubali</td>
      <td>No</td>
      <td>19-07-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>38</th>
      <td>Karthik</td>
      <td>Male</td>
      <td>24.0</td>
      <td>70.0</td>
      <td>2.000000</td>
      <td>62</td>
      <td>6.0</td>
      <td>2024-07-21</td>
      <td>Belagavi</td>
      <td>Yes</td>
      <td>2024-07-21</td>
      <td>2024-07-21</td>
    </tr>
    <tr>
      <th>39</th>
      <td>Divakar</td>
      <td>Male</td>
      <td>22.0</td>
      <td>75.0</td>
      <td>2.500000</td>
      <td>66</td>
      <td>NaN</td>
      <td>21/07/2024</td>
      <td>Tumkur</td>
      <td>Yes</td>
      <td>21-07-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>40</th>
      <td>Nisha</td>
      <td>Female</td>
      <td>21.0</td>
      <td>93.0</td>
      <td>4.500000</td>
      <td>90</td>
      <td>10.0</td>
      <td>2024/07/23</td>
      <td>Shivamoga</td>
      <td>No</td>
      <td>2024-07-23</td>
      <td>2024-07-23</td>
    </tr>
    <tr>
      <th>41</th>
      <td>Suraj</td>
      <td>Male</td>
      <td>20.0</td>
      <td>68.0</td>
      <td>1.500000</td>
      <td>57</td>
      <td>5.0</td>
      <td>23-07-2024</td>
      <td>Mangaluru</td>
      <td>Yes</td>
      <td>23-07-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>42</th>
      <td>Keerthi</td>
      <td>Female</td>
      <td>22.0</td>
      <td>86.0</td>
      <td>3.500000</td>
      <td>83</td>
      <td>9.0</td>
      <td>2024.07.25</td>
      <td>Udupi</td>
      <td>No</td>
      <td>2024.07.25</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>43</th>
      <td>Manjunath</td>
      <td>Male</td>
      <td>25.0</td>
      <td>62.0</td>
      <td>1.000000</td>
      <td>49</td>
      <td>4.0</td>
      <td>25/07/2024</td>
      <td>Bengaluru</td>
      <td>Yes</td>
      <td>25-07-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>44</th>
      <td>Pavithra</td>
      <td>Female</td>
      <td>20.0</td>
      <td>89.0</td>
      <td>4.000000</td>
      <td>86</td>
      <td>9.0</td>
      <td>2024/07/27</td>
      <td>Mysore</td>
      <td>No</td>
      <td>2024-07-27</td>
      <td>2024-07-27</td>
    </tr>
    <tr>
      <th>45</th>
      <td>Rakesh</td>
      <td>Female</td>
      <td>23.0</td>
      <td>81.0</td>
      <td>3.000000</td>
      <td>74</td>
      <td>8.0</td>
      <td>27-07-2024</td>
      <td>Dharwad</td>
      <td>Yes</td>
      <td>27-07-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>46</th>
      <td>Sahana</td>
      <td>Female</td>
      <td>21.0</td>
      <td>95.0</td>
      <td>5.000000</td>
      <td>93</td>
      <td>10.0</td>
      <td>2024-07-29</td>
      <td>Hubali</td>
      <td>No</td>
      <td>2024-07-29</td>
      <td>2024-07-29</td>
    </tr>
    <tr>
      <th>47</th>
      <td>Tejas</td>
      <td>Male</td>
      <td>22.0</td>
      <td>76.0</td>
      <td>2.500000</td>
      <td>65</td>
      <td>7.0</td>
      <td>29/07/2024</td>
      <td>Belgaum</td>
      <td>Yes</td>
      <td>29-07-2024</td>
      <td>NaT</td>
    </tr>
    <tr>
      <th>48</th>
      <td>Arjun</td>
      <td>Male</td>
      <td>22.0</td>
      <td>89.0</td>
      <td>2.500000</td>
      <td>69</td>
      <td>7.0</td>
      <td>2024-06-22</td>
      <td>Bengaluru</td>
      <td>Yes</td>
      <td>2024-06-22</td>
      <td>2024-06-22</td>
    </tr>
    <tr>
      <th>49</th>
      <td>Swathi</td>
      <td>Female</td>
      <td>20.0</td>
      <td>91.0</td>
      <td>4.500000</td>
      <td>94</td>
      <td>10.0</td>
      <td>07/07/2024</td>
      <td>Belgaum</td>
      <td>No</td>
      <td>07-07-2024</td>
      <td>NaT</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python

```


```python
data={
    "Employee_id" : [101,102,103,104,105],
    "Name" : ["Rahul", "Priya", "Arjun", "Sneha", "Kiran"],
    "Department" : ["IT","HR","IT","Finance","HR"],
    "Gender" : ["Male","Female","Male","Female","Male"],
    "Experience" : ["2 years","5 years","3 years","7 years","1 years"],
    "Performance" : ["Excellent","Good","Average","Excellent","Good"],
    "Joining_Date" : ["2022-06-15","2020-03-10","2021-08-20","2018-01-25","2023-07-01"],
}
```


```python
df=pd.DataFrame(data)
print(df.head())
```

       Employee_id   Name Department  Gender Experience Performance Joining_Date
    0          101  Rahul         IT    Male    2 years   Excellent   2022-06-15
    1          102  Priya         HR  Female    5 years        Good   2020-03-10
    2          103  Arjun         IT    Male    3 years     Average   2021-08-20
    3          104  Sneha    Finance  Female    7 years   Excellent   2018-01-25
    4          105  Kiran         HR    Male    1 years        Good   2023-07-01
    


```python
df["Experience"]=df["Experience"].str.extract(r"(\d+)").astype(int)
```


```python
print(df.head(20))
```

       Employee_id   Name Department  Gender  Experience Performance Joining_Date
    0          101  Rahul         IT    Male           2   Excellent   2022-06-15
    1          102  Priya         HR  Female           5        Good   2020-03-10
    2          103  Arjun         IT    Male           3     Average   2021-08-20
    3          104  Sneha    Finance  Female           7   Excellent   2018-01-25
    4          105  Kiran         HR    Male           1        Good   2023-07-01
    


```python
df["Joining_Year"] = df["Join_Date"].dt.year
print(df["Joining_Year"])
```


    ---------------------------------------------------------------------------

    KeyError                                  Traceback (most recent call last)

    File ~\PyCharmMiscProject\.venv\Lib\site-packages\pandas\core\indexes\base.py:3641, in Index.get_loc(self, key)
       3640 try:
    -> 3641     return self._engine.get_loc(casted_key)
       3642 except KeyError as err:
    

    File pandas/_libs/index.pyx:168, in pandas._libs.index.IndexEngine.get_loc()
    --> 168 'Could not get source, probably due dynamically evaluated source code.'
    

    File pandas/_libs/index.pyx:197, in pandas._libs.index.IndexEngine.get_loc()
    --> 197 'Could not get source, probably due dynamically evaluated source code.'
    

    File pandas/_libs/hashtable_class_helper.pxi:7668, in pandas._libs.hashtable.PyObjectHashTable.get_item()
    -> 7668 'Could not get source, probably due dynamically evaluated source code.'
    

    File pandas/_libs/hashtable_class_helper.pxi:7676, in pandas._libs.hashtable.PyObjectHashTable.get_item()
    -> 7676 'Could not get source, probably due dynamically evaluated source code.'
    

    KeyError: 'Join_Date'

    
    The above exception was the direct cause of the following exception:
    

    KeyError                                  Traceback (most recent call last)

    Cell In[150], line 1
    ----> 1 df["Joining_Year"] = df["Join_Date"].dt.year
          2 print(df["Joining_Year"])
    

    File ~\PyCharmMiscProject\.venv\Lib\site-packages\pandas\core\frame.py:4378, in DataFrame.__getitem__(self, key)
       4374 
       4375         if is_single_key:
       4376             if self.columns.nlevels > 1:
       4377                 return self._getitem_multilevel(key)
    -> 4378             indexer = self.columns.get_loc(key)
       4379             if is_integer(indexer):
       4380                 indexer = [indexer]
       4381         else:
    

    File ~\PyCharmMiscProject\.venv\Lib\site-packages\pandas\core\indexes\base.py:3648, in Index.get_loc(self, key)
       3643     if isinstance(casted_key, slice) or (
       3644         isinstance(casted_key, abc.Iterable)
       3645         and any(isinstance(x, slice) for x in casted_key)
       3646     ):
       3647         raise InvalidIndexError(key) from err
    -> 3648     raise KeyError(key) from err
       3649 except TypeError:
       3650     # If we have a listlike key, _check_indexing_error will raise
       3651     #  InvalidIndexError. Otherwise we fall through and re-raise
       3652     #  the TypeError.
       3653     self._check_indexing_error(key)
    

    KeyError: 'Join_Date'



```python

```


```python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
```


```python
data = {
    "Student": ["Asha", "Rahul", "Vikram", "Sneha", "Kiran"],
    "Gender": ["Female", "Male", "Male", "Female", "Male"],
    "Course": ["Python", "Java", "Python", "Java", "C++"],
    "Age": [20, 21, 19, 22, 20],
    "Study_Hours": ["3 hours", "5 hours", "2 hours", "6 hours", "4 hours"],
    "Attendance": ["85%", "92%", "68%", "95%", "78%"],
    "Exam_Date": [
        "2026-09-10",
        "2026-09-10",
        "2026-09-11",
        "2026-09-11",
        "2026-09-12"
    ],
    "Feedback": [
        "Good understanding of Python",
        "Excellent Java programming skills",
        "Needs improvement in programming",
        "Excellent performance in Java",
        "Good but needs more practice"
    ],
    "Result": ["Pass", "Pass", "Fail", "Pass", "Pass"]
}

```


```python
df = pd.DataFrame(data)
print(df)
```

      Student  Gender  Course  Age Study_Hours Attendance   Exam_Date  \
    0    Asha  Female  Python   20     3 hours        85%  2026-09-10   
    1   Rahul    Male    Java   21     5 hours        92%  2026-09-10   
    2  Vikram    Male  Python   19     2 hours        68%  2026-09-11   
    3   Sneha  Female    Java   22     6 hours        95%  2026-09-11   
    4   Kiran    Male     C++   20     4 hours        78%  2026-09-12   
    
                                Feedback Result  
    0       Good understanding of Python   Pass  
    1  Excellent Java programming skills   Pass  
    2   Needs improvement in programming   Fail  
    3      Excellent performance in Java   Pass  
    4       Good but needs more practice   Pass  
    


```python
data_1 = {
    "Customer": ["C001", "C002", "C003", "C004", "C005"],
    "Gender": ["Male", "Female", "Female", "Male", "Female"],
    "Device": ["Mobile", "Laptop", "Tablet", "Mobile", "Laptop"],
    "Age": [25, 32, 28, 40, 23],
    "Time_Spent": [
        "15 minutes",
        "30 minutes",
        "10 minutes",
        "45 minutes",
        "20 minutes"
    ],
    "Purchase_Amount": [
        "₹1500",
        "₹3200",
        "₹0",
        "₹5400",
        "₹2100"
    ],
    "Visit_Date": [
        "2026-09-01",
        "2026-09-02",
        "2026-09-03",
        "2026-09-03",
        "2026-09-04"
    ],
    "Review": [
        "Very good product",
        "Excellent quality and fast delivery",
        "Product was not useful",
        "Amazing product and service",
        "Good quality"
    ],
    "Purchased": ["Yes", "Yes", "No", "Yes", "Yes"]
}

```


```python
df_1 = pd.DataFrame(data_1)
print(df_1)
```

      Customer  Gender  Device  Age  Time_Spent Purchase_Amount  Visit_Date  \
    0     C001    Male  Mobile   25  15 minutes           ₹1500  2026-09-01   
    1     C002  Female  Laptop   32  30 minutes           ₹3200  2026-09-02   
    2     C003  Female  Tablet   28  10 minutes              ₹0  2026-09-03   
    3     C004    Male  Mobile   40  45 minutes           ₹5400  2026-09-03   
    4     C005  Female  Laptop   23  20 minutes           ₹2100  2026-09-04   
    
                                    Review Purchased  
    0                    Very good product       Yes  
    1  Excellent quality and fast delivery       Yes  
    2               Product was not useful        No  
    3          Amazing product and service       Yes  
    4                         Good quality       Yes  
    


```python
df["Study_Hours"] = df["Study_Hours"].str.extract(r"(\d+)").astype(int)
print(df.head())
```

      Student  Gender  Course  Age  Study_Hours Attendance   Exam_Date  \
    0    Asha  Female  Python   20            3        85%  2026-09-10   
    1   Rahul    Male    Java   21            5        92%  2026-09-10   
    2  Vikram    Male  Python   19            2        68%  2026-09-11   
    3   Sneha  Female    Java   22            6        95%  2026-09-11   
    4   Kiran    Male     C++   20            4        78%  2026-09-12   
    
                                Feedback Result  
    0       Good understanding of Python   Pass  
    1  Excellent Java programming skills   Pass  
    2   Needs improvement in programming   Fail  
    3      Excellent performance in Java   Pass  
    4       Good but needs more practice   Pass  
    


```python
df["Attendance"] = df["Attendance"].str.replace("%","").astype(int)
print(df.head())
```

      Student  Gender  Course  Age  Study_Hours  Attendance   Exam_Date  \
    0    Asha  Female  Python   20            3          85  2026-09-10   
    1   Rahul    Male    Java   21            5          92  2026-09-10   
    2  Vikram    Male  Python   19            2          68  2026-09-11   
    3   Sneha  Female    Java   22            6          95  2026-09-11   
    4   Kiran    Male     C++   20            4          78  2026-09-12   
    
                                Feedback Result  
    0       Good understanding of Python   Pass  
    1  Excellent Java programming skills   Pass  
    2   Needs improvement in programming   Fail  
    3      Excellent performance in Java   Pass  
    4       Good but needs more practice   Pass  
    


```python
df["Gender"] = df["Gender"].str.replace({"Male":"0","Female":"1"}).astype(int)
print(df.head())
```

      Student  Gender  Course  Age  Study_Hours  Attendance   Exam_Date  \
    0    Asha       1  Python   20            3          85  2026-09-10   
    1   Rahul       0    Java   21            5          92  2026-09-10   
    2  Vikram       0  Python   19            2          68  2026-09-11   
    3   Sneha       1    Java   22            6          95  2026-09-11   
    4   Kiran       0     C++   20            4          78  2026-09-12   
    
                                Feedback Result  
    0       Good understanding of Python   Pass  
    1  Excellent Java programming skills   Pass  
    2   Needs improvement in programming   Fail  
    3      Excellent performance in Java   Pass  
    4       Good but needs more practice   Pass  
    


```python
encode = LabelEncoder()
df["Gender_Encode"] = encode.fit_transform(df["Gender"])
print(df.head())
```

      Student  Gender  Course  Age  Study_Hours  Attendance   Exam_Date  \
    0    Asha  Female  Python   20            3          85  2026-09-10   
    1   Rahul    Male    Java   21            5          92  2026-09-10   
    2  Vikram    Male  Python   19            2          68  2026-09-11   
    3   Sneha  Female    Java   22            6          95  2026-09-11   
    4   Kiran    Male     C++   20            4          78  2026-09-12   
    
                                Feedback Result  Gender_Encode  
    0       Good understanding of Python   Pass              0  
    1  Excellent Java programming skills   Pass              1  
    2   Needs improvement in programming   Fail              1  
    3      Excellent performance in Java   Pass              0  
    4       Good but needs more practice   Pass              1  
    


```python
print(get_dummies(df["Course"]))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    Cell In[60], line 1
    ----> 1 print(get_dummies(df["Course"]))
    

    NameError: name 'get_dummies' is not defined



```python
df["Tokens"] = df["Feedback"].apply(lambda x: x.lower().split())
df[["Feedback","Tokens"]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feedback</th>
      <th>Tokens</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Good understanding of Python</td>
      <td>[good, understanding, of, python]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Excellent Java programming skills</td>
      <td>[excellent, java, programming, skills]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Needs improvement in programming</td>
      <td>[needs, improvement, in, programming]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Excellent performance in Java</td>
      <td>[excellent, performance, in, java]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Good but needs more practice</td>
      <td>[good, but, needs, more, practice]</td>
    </tr>
  </tbody>
</table>
</div>




```python
df["Tokens_1"] = df["Feedback"].str.split()
df[["Feedback","Tokens","Tokens_1"]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Feedback</th>
      <th>Tokens</th>
      <th>Tokens_1</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Good understanding of Python</td>
      <td>[good, understanding, of, python]</td>
      <td>[Good, understanding, of, Python]</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Excellent Java programming skills</td>
      <td>[excellent, java, programming, skills]</td>
      <td>[Excellent, Java, programming, skills]</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Needs improvement in programming</td>
      <td>[needs, improvement, in, programming]</td>
      <td>[Needs, improvement, in, programming]</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Excellent performance in Java</td>
      <td>[excellent, performance, in, java]</td>
      <td>[Excellent, performance, in, Java]</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Good but needs more practice</td>
      <td>[good, but, needs, more, practice]</td>
      <td>[Good, but, needs, more, practice]</td>
    </tr>
  </tbody>
</table>
</div>




```python
Counter_1 = CountVectorizer()
x = Counter_1.fit_transform(df["Feedback"])
print(Counter_1.get_feature_names_out())
print(x.toarray())
```

    ['but' 'excellent' 'good' 'improvement' 'in' 'java' 'more' 'needs' 'of'
     'performance' 'practice' 'programming' 'python' 'skills' 'understanding']
    [[0 0 1 0 0 0 0 0 1 0 0 0 1 0 1]
     [0 1 0 0 0 1 0 0 0 0 0 1 0 1 0]
     [0 0 0 1 1 0 0 1 0 0 0 1 0 0 0]
     [0 1 0 0 1 1 0 0 0 1 0 0 0 0 0]
     [1 0 1 0 0 0 1 1 0 0 1 0 0 0 0]]
    


```python
df["Result_updated"] = df["Result"].map({"Pass":"1", "Fail":"0"})
df["Result_updated"]
```




    0    1
    1    1
    2    0
    3    1
    4    1
    Name: Result_updated, dtype: str




```python
df["Attendance_binary"] = (df["Attendance"] >= 75).astype(int)
df[["Attendance_binary","Attendance"]]
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Attendance_binary</th>
      <th>Attendance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>85</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>92</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>68</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1</td>
      <td>95</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1</td>
      <td>78</td>
    </tr>
  </tbody>
</table>
</div>




```python

```


```python
import numpy as np
```


```python
image = np.array([
    [0,0,255,255,0],
    [0,255,0,255,0],
    [0,255,255,255,0],
    [0,0,255,0,0],
    [0,0,255,0,0]
])

print(image)
```

    [[  0   0 255 255   0]
     [  0 255   0 255   0]
     [  0 255 255 255   0]
     [  0   0 255   0   0]
     [  0   0 255   0   0]]
    


```python
image_binary = ( image >= 75).astype(int)
print(image_binary)
```

    [[0 0 1 1 0]
     [0 1 0 1 0]
     [0 1 1 1 0]
     [0 0 1 0 0]
     [0 0 1 0 0]]
    


```python
plt.imshow(image)
```




    <matplotlib.image.AxesImage at 0x2d9591a1610>




    
![png](Student_files/Student_68_1.png)
    



```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LinearRegression
```


```python
import pandas as pd

df = pd.read_csv(r"C:\Users\Ethnotech\Downloads\employee_supervised_learning_cleaned2.csv")


```


```python
df.head(500)
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>education_level</th>
      <th>department</th>
      <th>experience_years</th>
      <th>performance_score</th>
      <th>attendance_pct</th>
      <th>training_hours</th>
      <th>projects_completed</th>
      <th>overtime_hours</th>
      <th>leadership_score</th>
      <th>job_satisfaction</th>
      <th>current_salary</th>
      <th>promoted</th>
      <th>next_year_salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>Bachelor</td>
      <td>HR</td>
      <td>0.0</td>
      <td>6.7</td>
      <td>92.2</td>
      <td>64.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>5.3</td>
      <td>8.1</td>
      <td>428000.0</td>
      <td>0.0</td>
      <td>583000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>36.0</td>
      <td>Master</td>
      <td>Sales</td>
      <td>14.0</td>
      <td>7.0</td>
      <td>82.7</td>
      <td>73.0</td>
      <td>10.0</td>
      <td>24.0</td>
      <td>5.4</td>
      <td>10.0</td>
      <td>1361000.0</td>
      <td>0.0</td>
      <td>1669000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>55.0</td>
      <td>Diploma</td>
      <td>IT</td>
      <td>30.0</td>
      <td>5.9</td>
      <td>86.9</td>
      <td>60.0</td>
      <td>19.0</td>
      <td>15.0</td>
      <td>5.4</td>
      <td>6.3</td>
      <td>1800000.0</td>
      <td>0.0</td>
      <td>2313000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>54.0</td>
      <td>Bachelor</td>
      <td>HR</td>
      <td>30.0</td>
      <td>8.0</td>
      <td>90.9</td>
      <td>41.0</td>
      <td>14.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>7.9</td>
      <td>1800000.0</td>
      <td>0.0</td>
      <td>2392000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42.0</td>
      <td>Master</td>
      <td>HR</td>
      <td>20.0</td>
      <td>5.4</td>
      <td>89.7</td>
      <td>28.0</td>
      <td>10.0</td>
      <td>14.0</td>
      <td>6.8</td>
      <td>5.8</td>
      <td>1695000.0</td>
      <td>0.0</td>
      <td>2074000.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>495</th>
      <td>37.0</td>
      <td>Bachelor</td>
      <td>Finance</td>
      <td>8.0</td>
      <td>8.8</td>
      <td>93.8</td>
      <td>49.0</td>
      <td>7.0</td>
      <td>26.0</td>
      <td>7.4</td>
      <td>9.4</td>
      <td>1082000.0</td>
      <td>0.0</td>
      <td>1329000.0</td>
    </tr>
    <tr>
      <th>496</th>
      <td>35.0</td>
      <td>Master</td>
      <td>Sales</td>
      <td>7.0</td>
      <td>3.9</td>
      <td>93.0</td>
      <td>28.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>3.8</td>
      <td>5.3</td>
      <td>865000.0</td>
      <td>0.0</td>
      <td>1043000.0</td>
    </tr>
    <tr>
      <th>497</th>
      <td>53.0</td>
      <td>Bachelor</td>
      <td>Finance</td>
      <td>25.0</td>
      <td>7.2</td>
      <td>91.8</td>
      <td>78.0</td>
      <td>18.0</td>
      <td>7.0</td>
      <td>6.9</td>
      <td>8.5</td>
      <td>1800000.0</td>
      <td>0.0</td>
      <td>2226000.0</td>
    </tr>
    <tr>
      <th>498</th>
      <td>37.0</td>
      <td>Master</td>
      <td>Finance</td>
      <td>12.0</td>
      <td>8.3</td>
      <td>88.0</td>
      <td>69.0</td>
      <td>12.0</td>
      <td>17.0</td>
      <td>7.2</td>
      <td>7.3</td>
      <td>1381000.0</td>
      <td>0.0</td>
      <td>1694000.0</td>
    </tr>
    <tr>
      <th>499</th>
      <td>54.0</td>
      <td>Bachelor</td>
      <td>HR</td>
      <td>28.0</td>
      <td>6.7</td>
      <td>90.6</td>
      <td>43.0</td>
      <td>19.0</td>
      <td>24.0</td>
      <td>5.6</td>
      <td>8.4</td>
      <td>1800000.0</td>
      <td>0.0</td>
      <td>2297000.0</td>
    </tr>
  </tbody>
</table>
<p>500 rows × 14 columns</p>
</div>




```python

```

    <class 'pandas.DataFrame'>
    RangeIndex: 502 entries, 0 to 501
    Data columns (total 14 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   age                 500 non-null    float64
     1   education_level     500 non-null    str    
     2   department          500 non-null    str    
     3   experience_years    500 non-null    float64
     4   performance_score   500 non-null    float64
     5   attendance_pct      500 non-null    float64
     6   training_hours      500 non-null    float64
     7   projects_completed  500 non-null    float64
     8   overtime_hours      500 non-null    float64
     9   leadership_score    500 non-null    float64
     10  job_satisfaction    500 non-null    float64
     11  current_salary      500 non-null    float64
     12  promoted            500 non-null    float64
     13  next_year_salary    500 non-null    float64
    dtypes: float64(12), str(2)
    memory usage: 55.0 KB
    


```python
df.drop_duplicates()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>education_level</th>
      <th>department</th>
      <th>experience_years</th>
      <th>performance_score</th>
      <th>attendance_pct</th>
      <th>training_hours</th>
      <th>projects_completed</th>
      <th>overtime_hours</th>
      <th>leadership_score</th>
      <th>job_satisfaction</th>
      <th>current_salary</th>
      <th>promoted</th>
      <th>next_year_salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>22.0</td>
      <td>Bachelor</td>
      <td>HR</td>
      <td>0.0</td>
      <td>6.7</td>
      <td>92.2</td>
      <td>64.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>5.3</td>
      <td>8.1</td>
      <td>428000.0</td>
      <td>0.0</td>
      <td>583000.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>36.0</td>
      <td>Master</td>
      <td>Sales</td>
      <td>14.0</td>
      <td>7.0</td>
      <td>82.7</td>
      <td>73.0</td>
      <td>10.0</td>
      <td>24.0</td>
      <td>5.4</td>
      <td>10.0</td>
      <td>1361000.0</td>
      <td>0.0</td>
      <td>1669000.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>55.0</td>
      <td>Diploma</td>
      <td>IT</td>
      <td>30.0</td>
      <td>5.9</td>
      <td>86.9</td>
      <td>60.0</td>
      <td>19.0</td>
      <td>15.0</td>
      <td>5.4</td>
      <td>6.3</td>
      <td>1800000.0</td>
      <td>0.0</td>
      <td>2313000.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>54.0</td>
      <td>Bachelor</td>
      <td>HR</td>
      <td>30.0</td>
      <td>8.0</td>
      <td>90.9</td>
      <td>41.0</td>
      <td>14.0</td>
      <td>4.0</td>
      <td>7.0</td>
      <td>7.9</td>
      <td>1800000.0</td>
      <td>0.0</td>
      <td>2392000.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>42.0</td>
      <td>Master</td>
      <td>HR</td>
      <td>20.0</td>
      <td>5.4</td>
      <td>89.7</td>
      <td>28.0</td>
      <td>10.0</td>
      <td>14.0</td>
      <td>6.8</td>
      <td>5.8</td>
      <td>1695000.0</td>
      <td>0.0</td>
      <td>2074000.0</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>496</th>
      <td>35.0</td>
      <td>Master</td>
      <td>Sales</td>
      <td>7.0</td>
      <td>3.9</td>
      <td>93.0</td>
      <td>28.0</td>
      <td>7.0</td>
      <td>0.0</td>
      <td>3.8</td>
      <td>5.3</td>
      <td>865000.0</td>
      <td>0.0</td>
      <td>1043000.0</td>
    </tr>
    <tr>
      <th>497</th>
      <td>53.0</td>
      <td>Bachelor</td>
      <td>Finance</td>
      <td>25.0</td>
      <td>7.2</td>
      <td>91.8</td>
      <td>78.0</td>
      <td>18.0</td>
      <td>7.0</td>
      <td>6.9</td>
      <td>8.5</td>
      <td>1800000.0</td>
      <td>0.0</td>
      <td>2226000.0</td>
    </tr>
    <tr>
      <th>498</th>
      <td>37.0</td>
      <td>Master</td>
      <td>Finance</td>
      <td>12.0</td>
      <td>8.3</td>
      <td>88.0</td>
      <td>69.0</td>
      <td>12.0</td>
      <td>17.0</td>
      <td>7.2</td>
      <td>7.3</td>
      <td>1381000.0</td>
      <td>0.0</td>
      <td>1694000.0</td>
    </tr>
    <tr>
      <th>499</th>
      <td>54.0</td>
      <td>Bachelor</td>
      <td>HR</td>
      <td>28.0</td>
      <td>6.7</td>
      <td>90.6</td>
      <td>43.0</td>
      <td>19.0</td>
      <td>24.0</td>
      <td>5.6</td>
      <td>8.4</td>
      <td>1800000.0</td>
      <td>0.0</td>
      <td>2297000.0</td>
    </tr>
    <tr>
      <th>500</th>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
<p>501 rows × 14 columns</p>
</div>




```python
df.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>age</th>
      <th>experience_years</th>
      <th>performance_score</th>
      <th>attendance_pct</th>
      <th>training_hours</th>
      <th>projects_completed</th>
      <th>overtime_hours</th>
      <th>leadership_score</th>
      <th>job_satisfaction</th>
      <th>current_salary</th>
      <th>promoted</th>
      <th>next_year_salary</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>500.000000</td>
      <td>5.000000e+02</td>
      <td>500.000000</td>
      <td>5.000000e+02</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>38.920000</td>
      <td>14.624000</td>
      <td>7.296800</td>
      <td>90.844600</td>
      <td>45.848000</td>
      <td>10.900000</td>
      <td>10.396000</td>
      <td>6.202400</td>
      <td>7.487600</td>
      <td>1.326438e+06</td>
      <td>0.010000</td>
      <td>1.677608e+06</td>
    </tr>
    <tr>
      <th>std</th>
      <td>10.059223</td>
      <td>9.667137</td>
      <td>1.315938</td>
      <td>4.750546</td>
      <td>17.176815</td>
      <td>5.592186</td>
      <td>6.828796</td>
      <td>1.158758</td>
      <td>1.212876</td>
      <td>4.864168e+05</td>
      <td>0.099598</td>
      <td>6.040924e+05</td>
    </tr>
    <tr>
      <th>min</th>
      <td>21.000000</td>
      <td>0.000000</td>
      <td>3.300000</td>
      <td>75.900000</td>
      <td>5.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.000000</td>
      <td>3.400000</td>
      <td>2.850000e+05</td>
      <td>0.000000</td>
      <td>4.320000e+05</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>30.000000</td>
      <td>6.000000</td>
      <td>6.400000</td>
      <td>87.800000</td>
      <td>34.000000</td>
      <td>6.000000</td>
      <td>5.000000</td>
      <td>5.475000</td>
      <td>6.700000</td>
      <td>8.962500e+05</td>
      <td>0.000000</td>
      <td>1.163750e+06</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>39.000000</td>
      <td>15.000000</td>
      <td>7.300000</td>
      <td>90.900000</td>
      <td>46.000000</td>
      <td>11.000000</td>
      <td>10.000000</td>
      <td>6.200000</td>
      <td>7.500000</td>
      <td>1.474000e+06</td>
      <td>0.000000</td>
      <td>1.833000e+06</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>48.000000</td>
      <td>23.000000</td>
      <td>8.200000</td>
      <td>94.200000</td>
      <td>57.000000</td>
      <td>16.000000</td>
      <td>15.000000</td>
      <td>7.000000</td>
      <td>8.400000</td>
      <td>1.800000e+06</td>
      <td>0.000000</td>
      <td>2.255000e+06</td>
    </tr>
    <tr>
      <th>max</th>
      <td>55.000000</td>
      <td>30.000000</td>
      <td>10.000000</td>
      <td>100.000000</td>
      <td>100.000000</td>
      <td>20.000000</td>
      <td>33.000000</td>
      <td>10.000000</td>
      <td>10.000000</td>
      <td>1.800000e+06</td>
      <td>1.000000</td>
      <td>2.440000e+06</td>
    </tr>
  </tbody>
</table>
</div>




```python
missing = df.isnull().sum()
print(missing)
```

    age                   2
    education_level       2
    department            2
    experience_years      2
    performance_score     2
    attendance_pct        2
    training_hours        2
    projects_completed    2
    overtime_hours        2
    leadership_score      2
    job_satisfaction      2
    current_salary        2
    promoted              2
    next_year_salary      2
    dtype: int64
    


```python
k_means()
```


```python

```


```python

```
