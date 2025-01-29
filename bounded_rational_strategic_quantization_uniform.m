function main
clc;
close all;
clear all;
 
%% parameters
sigma_xsq=1; % variance of noiseless source U
sigma_thsq=1; % variance theta
Mval=[4];
% U theta W range
at=0;
bt=1;
a=0;
b=1;
f1= 1/(bt-at)*(1/(b-a)); % pdf of X conditional on theta
fx= (1/(b-a));
endistM=zeros(length(Mval),1);
dedistM=zeros(length(Mval),1);

%%
[xmall,ymall,Mval1]=max_quant();
Klevel=3;
pt=zeros(1,Klevel);
lambdaval=[0.001 0.01 0.1 1 4 10 100 700];




for lambda=lambdaval
for kind=1:Klevel
pt(kind)=exp(-lambda)*lambda^(kind-1)/factorial(kind-1);
end
pt=pt./sum(pt);
for M=Mval
xthetaval=zeros(Klevel,M+1);
xq=[a xmall(find(M==Mval1),1:M-1) b];
xthetaval(1,:)=xq;
ym=reconstruction_nonstr(xq,fx,a,b); % reconstruction levels
[encoder_dist_level0] = encoderdistortion_nonstr(xq,a,b,fx);
[dist_dec_level0]=decoderdistortion_nonstr(xq,ym,fx);
ym_non=ym;
save(strcat('lambda',num2str(lambda),'_level_',num2str(0),'_M',num2str(M),'Xthetarho',num2str(rho),'_varth',num2str(sigma_thsq),'varx',num2str(sigma_xsq),'data.mat'),'xthetaval','xq','encoder_dist_level0','dist_dec_level0','ym','sigma_thsq','sigma_xsq');
%level 1
for levelind=2:Klevel
xq=[a (ym(1:end-1)+ym(2:end))/2 b];
[encoder_dist_levelk] = encoderdistortion(xq,thval,a,b,f1,pth,ym);
[dist_dec_levelk]=decoderdistortion_levelk(xthetaval,ym,f1,pth,a,b,pt,levelind);

xthetaval(levelind,:)=xq;
save(strcat('lambda',num2str(lambda),'_level_',num2str(levelind-1),'_M',num2str(M),'Xthetauniform_varth',num2str(sigma_thsq),'varx',num2str(sigma_xsq),'data.mat'),'xthetaval','xq','encoder_dist_levelk','dist_dec_levelk','ym','sigma_thsq','sigma_xsq','levelind');
ym=reconstruction(xthetaval,thval,f1,pth,a,b,pt,levelind);
end
ym=reconstruction(xthetaval,thval,f1,pth,a,b,pt,Klevel+1);
[dist_dec_act]=decoderdistortion_levelk(xthetaval,ym,f1,pth,a,b,pt,levelind+1);
save(strcat('lambda',num2str(lambda),'_N',num2str(Klevel),'_M',num2str(M),'Xthetarho_varth',num2str(sigma_thsq),'varx',num2str(sigma_xsq),'data.mat'),'xthetaval','xq','dist_dec_act','ym','sigma_thsq','sigma_xsq');
end
end


function [dist_dec]=decoderdistortion_nonstr(xq,ym,fx)
M=length(xq)-1;
dist_dec=0;
for i=1:M
    fux2=@(xv) (xv-ym(i)).^2.*fx(xv);
    dist_dec=dist_dec+integral(fux2,xq(i),xq(i+1));
end


function [dist_dec]=decoderdistortion_levelk(xthetaval,ym,f1,pth,a,b,pt,ind)
pt=pt(1:ind-1)/sum(pt(1:ind-1));
M=size(xthetaval,3)-1;
dist_dec=0;
for ptind=1:ind-1
for i=1:M
        fux2=@(tv,xv) (xv-ym(i)).^2.*f1(xv,tv);
        ymin=@(tv) xthetaval(ptind,i)-tv;
        ymax=@(tv) xthetaval(ptind,i+1)-tv;
        dist_dec=dist_dec+pt(ptind)*integral2(fux2,at,bt,ymin,ymax,'Method','tiled');
end
end

% function [dist_dec]=decoderdistortion(xthetam,ym,fx,pth,a,b)
% M=size(xthetam,2)-1;
% dist_dec=0;
% for i=1:M
%     for k=1:length(pth)
%         fux2=@(xv) (xv-ym(i)).^2.*fx(xv)*pth(k);
%         dist_dec=dist_dec+integral(fux2,xthetam(k,i),xthetam(k,i+1));
%     end
% end

function [ym]=reconstruction_nonstr(xq,fx,a,b)
M=size(xq,2)-1;
ym=zeros(1,M);
for i=1:M
    fux1= @(xv) xv.*fx(xv);
    num=integral(fux1,xq(i),xq(i+1));
    den=integral(fx,xq(i),xq(i+1));
    if den~=0
    ym(i)=num/den;
    end
end 

function [ym]=reconstruction(xthetaval,thval,f1,pth,a,b,pt,ind)
pt=pt(1:ind-1)/sum(pt(1:ind-1));
M=size(xthetaval,3)-1;
ym=zeros(1,M);
for i=1:M
    num=0;
    den=0;
    for k=1:ind-1
    for j=1:length(thval)
        fux1= @(xv) xv.*f1;
        ymin=@(tv) xthetaval(k,i)-tv;
        ymax=@(tv) xthetaval(k,i+1)-tv;
        num=num+pt(k)*integral(fux1,at,bt,ymin,ymax);
        den=den+pt(k)*integral( f1,at,bt,ymin,ymax);
    end
    end
    if den~=0
    ym(i)=num/den;
    end
end

function [f22] = encoderdistortion_nonstr(xq,a,b,fx)
M=length(xq)-1;
[ym]=reconstruction_nonstr(xq,fx,a,b);
f22=0;
for i=1:M
    f22=f22+integral(@(xv)(xv-ym(i)).^2.*fx,xq(i),xq(i+1));
end


function [f22] = encoderdistortion(x,thval,a,b,f1,pth,ym)
M=size(x,2)-1;
f22=0;
for i=1:M
        ymin=@(tv) x(i)-tv;
        ymax=@(tv) x(i+1)-tv;
        f22=f22+integral2(@(xv)(xv+tv-ym(i)).^2.*f1,at,bt,ymin,ymax);
end

function [xmall,ymall,Mval]=max_quant() 
%max quantization table from...
%zero mean, variance 1 gaussian
Mval=[2 4 8 16 32];
xmall=zeros(length(Mval),max(Mval)-1);
ymall=zeros(length(Mval),max(Mval));
M=2;
xmall(find(M==Mval),1:M-1)=0;
ymall(find(M==Mval),1:M)=[-0.7980 0.7980];
M=4;
temp=[0.9816];
xmall(find(M==Mval),1:M-1)=[-flip(temp) 0 temp];
temp=[0.4528 1.510];
ymall(find(M==Mval),1:M)=[-flip(temp) temp];
M=8;
temp=[0.5006 1.050 1.748];
xmall(find(M==Mval),1:M-1)=[-flip(temp) 0 temp];
temp=[0.2451 0.7560 1.344 2.152];
ymall(find(M==Mval),1:M)=[-flip(temp) temp];
M=16;
temp=[0.2582 0.5224 0.7996 1.099 1.437 1.844 2.401];
xmall(find(M==Mval),1:M-1)=[-flip(temp) 0 temp];
temp=[0.1284 0.3881 0.6568 0.9424 1.256 1.618 2.069 2.733];
ymall(find(M==Mval),1:M)=[-flip(temp) temp];
M=32;
temp=[0.1320 0.2648 0.3991 0.5359 0.6761 0.8210 0.9718 1.130 1.299 1.482 1.682 1.908 2.174 2.505 2.977];
xmall(find(M==Mval),1:M-1)=[-flip(temp) 0 temp];
temp=[0.06590 0.1981 0.3314 0.4668 0.6050 0.7473 0.8947 1.049 1.212 1.387 1.577 1.788 2.029 2.319 2.692 3.263];
ymall(find(M==Mval),1:M)=[-flip(temp) temp];


