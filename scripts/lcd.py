import IPython
import numpy as np
import utils
import encoding

def LCD(data):
	assert len(data.shape) == 3
	M = data.shape[0]
	a = data.shape[1] 
	assert a == data.shape[2]
	lcd = np.swapaxes(data, 0, 2)
	lcd = lcd.reshape(a * a, M)
	assert lcd.shape[0] == a * a
	assert lcd.shape[1] == M
	return lcd