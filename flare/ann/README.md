ann
===

# metrics


| metric | full name          | formula                                                | similarity | requirements |
| ------ | ------------------ | ------------------------------------------------------ | ---------- | ------------ |
| l1     | manhattan distance | $\sum_{i = 0}^n { \lvert x_i-y_i\rvert }$              | smaller    | none         |
| l2     | Euclid             | $\sqrt{\sum_{i = 0}^n {(x_i-y_i)^2}}$       | smaller    | none         |
| ip     | inner product      | $\sum_{i=0}^n x_i*y_i$                                 | bigger     | none         |
| cosine | cosine             | $\frac{\sqrt{\sum_{i = 0}^n { \lvert x_i-y_i\rvert }}} {\sqrt{\sum_{i = 0}^n { \lvert x_i-y_i\rvert }}}$|            |              |
