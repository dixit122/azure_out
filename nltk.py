import collections.abc
#hyper needs the four following aliases to be done manually.
collections.Iterable = collections.abc.Iterable
collections.Mapping = collections.abc.Mapping
collections.MutableSet = collections.abc.MutableSet
collections.MutableMapping = collections.abc.MutableMapping
#Now import hyper
import hyper

from inltk.inltk import setup

setup('hi')

from inltk.inltk import tokenize

text = "भारत के स्वतंत्रता सेनानी और बापू के तौर पर अपनी पहचान बनाने वाले मोहनदास करमचंद गांधी का जन्म 2 अक्टूबर 1869 को गुजरात के पोरबंदर में हुआ था। उन्होंने अंग्रेज़ों की गुलामी से भारत को आज़ाद कराने के लिए अपना पूरा जीवन दे दिया था। आज़ादी के लिए उन्होंने चंपारण, खेड़ा, आंदोलन, आंदोलन और भारत छोड़ो आदि आंदोलन किए।"
x = tokenize(text ,'hi')

print(x)

from inltk.inltk import predict_next_words

predict_next_words("भारत के स्वतंत्रता सेनानी और" , 3, 'hi')
