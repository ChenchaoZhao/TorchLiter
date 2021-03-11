import liter

def test_train_stub():
    
    train = liter.stub.Train('loader')
    
    assert train.dataloader == 'loader'
    
    assert 'iteration' in train.__dict__
    
    assert train.iteration == 0