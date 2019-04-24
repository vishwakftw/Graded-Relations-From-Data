# [Graded Relations from Data](https://github.com/vishwakftw/Graded-Relations-From-Data)

Implementation of the framework in the paper:

Waegeman, W., Pahikkala, T., Airola, A., Salakoski, T., Stock, M., & De Baets, B. (2012). A kernel-based framework for learning graded relations from data. _IEEE Transactions on Fuzzy Systems, 20_(6), 1090-1101.

## Setup
### Prerequisites
* Python 3.6+
* pip (for setting up dependencies)

### Setting up dependencies
Use
```bash
$ pip install -r requirements.txt
```
### Setting up the package
Use
```bash
$ python setup.py install
```

## Running the experiments
The following describe the procedure to run the experiments in VI. A. and VI. C. (including generating simulated data) of the paper.

### VI. A.
Use
```bash
$ python scripts/experiment1.py [options]
```

To view the list of available options and their descriptions, use
```bash
$ python scripts/experiment1.py --help
```

### VI. C.
Use
```bash
$ python scripts/experiment2.py [options]
```

To view the list of available options and their descriptions, use
```bash
$ python scripts/experiment2.py --help
```

## License
This code is provided using the [MIT License](LICENSE).

---
This project was a part of the course MA6040: Fuzzy Logic Connectives: Theory and Applications, offered in Spring 2019 at IIT Hyderabad.

Team members: [Vishwak Srinivasan](https://vishwakftw.github.io) and [Sukrut Rao](https://sukrutrao.github.io).
