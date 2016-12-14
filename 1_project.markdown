---
layout: post
title: OpenCL on Cyclone V SoC
abstract: In this project OpenCL is implemented on an Altera® Cyclone® V System-on-Chip (SoC) development kit, where an ARM® Cortex®-A9 will serve as the host and a field-programmable gate array (FPGA) will serve as the device.
img: /img/project_1.png
---

This project utilizes C++ as the host language and OpenCL C as the device language. 


This theme implements a built-in Jekyll feature, the use of Pygments, for sytanx highlighting. It supports more than 100 languages. This example is in C++. All you have to do is wrap your code in a liquid tag: 
{% raw  %}
{% highlight c++ %}  <br/> code code code <br/> {% endhighlight %}{% endraw %}

Produces something like this: 

{% highlight c++ %}

int main(int argc, char const *argv[])
{
	string myString;

	cout << "input a string: ";
	getline(cin, myString);
	int length = myString.length();
	
	char charArray = new char * [length];

	charArray = myString;
	for(int i = 0; i < length; ++i){
		cout << charArray[i] << " ";
	}
	
	return 0;
}

{% endhighlight %}