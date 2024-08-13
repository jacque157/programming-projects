# Associative-array

This project is an C++ implementation of associative array (map/dictionary) with open addressing. This project also includes methods for writing and reading JSON files.

For addressing of values in associative array the so called closed hashing (open addressing) is used. The key-value doubles are represented by data structure Item. Instances of class Item are stored in DynamicArray data structure. These items are accessed via number IDs, which are generated using std::hash function. Associative arrays can be both written into and read from JSON files.

This implementation is trying to tackle the problem with special characters and nested sequences.
