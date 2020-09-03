# A Lightweight Chinese FAQ

A fast and efficient chinese FAQ system with One shot learning (Siamese) and TF-IDF techniques.


### requirements
* tensorflow  1.14.0
* keras   2.2.4


### usage

See more details in faq_siamese.py

```python
if __name__ == '__main__':
    # train model
    train()
    # show candidate
    print('*' * 30)
    exps = ['为什么植物会指南','小孩为什么不能吃补品中药？']
    for exp in exps:
        for x in ShowCandidate(exp):
            print(x)
        print('*'*30)
    # answer a question
    print(Answer('为何植物会吃石头'))
```