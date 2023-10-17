ann
===

# metrics


| metric  | full name          | formula                                                                                           | similarity | requirements |
|---------|--------------------|---------------------------------------------------------------------------------------------------|------------|--------------|
| l1      | manhattan distance | $\sum_{i = 0}^n { \lvert x_i-y_i\rvert }$                                                         | smaller    | none         |
| l2      | Euclid             | $\sqrt{\sum_{i = 0}^n {(x_i-y_i)^2}}$                                                             | smaller    | none         |
| ip      | inner product      | $\sum_{i=0}^n x_i*y_i$                                                                            | bigger     | none         |
| cosine  | cosine             | $1 - \frac{\sum_{i = 0}^n {x_i*y_i}} {\sqrt{\sum_{i = 0}^n {x_i}} {\sqrt{\sum_{i = 0}^n {y_i}}}}$ | smaller    | none         |
| jaccard | jaccard            | 1- sum (x(i) & y(i)) / sum(x(i) \| y(i) )                                                         | smaller          | integer      |
| hamming | hamming            | 1- sum (x(i) ^ y(i)))                                                         |    smaller              |    integer          |
