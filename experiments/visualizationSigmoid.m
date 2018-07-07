a=500;
shift=0.02
x=linspace(-1,1,1000);
y1=1./(1+exp(-(x-shift).*a));
y2=1./(1+exp(-(-x-shift).*a));
subplot(1,3,1)
plot(x,y1)
subplot(1,3,2)
plot(x,y2)
subplot(1,3,3)
plot(x,y1+y2)
figure 
plot(x,y1+y2)




