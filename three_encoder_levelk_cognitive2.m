function main
clc;
close all;
clear all;

%% parameters
lambdaval=[0.001 0.01 0.1 1 10 20 100 700];
lambdaval=[ 700 100 10 1 0.1 0.01 0.001];
%variance
sigma_thsqval=[  0.1 1.5 ]; % variance theta
sigma_xsq=1; % variance of source X
rhoval=[0]; %correlation
% rhoval=[-0.9  ]; 
M=8;

numsamp=[30 30 30 30];%number of samples on x for the grid in [-3,3]

rn=10; % number of initializations: form a grid then pick rn number of initializations randomly 

%%
% discretizing theta, theta in [at,bt], mean mut, variance sigma_thsq
at=-5;
bt=5;
% thval=linspace(at+(bt-at)/(2*nt),bt-(bt-at)/(2*nt),nt); 
mut=0;


for sigma_thsq=sigma_thsqval
thval1=linspace(at,mut-2*sigma_thsq,1);
thval2=linspace(mut-2*sigma_thsq,mut-sigma_thsq,5);
thval3=linspace(mut-sigma_thsq,mut+sigma_thsq,11);
thval4=linspace(mut+sigma_thsq,mut+2*sigma_thsq,5);
thval5=linspace(mut+2*sigma_thsq,bt,1);
thval=[thval1(2:end) thval2(2:end) thval3(2:end) thval4(2:end) thval5(2:end-1)];
thval=[thval1 thval2(2:end) thval3(2:end) thval4(2:end) thval5(2:end-1)]';
nt=length(thval);
% pdf of theta
pth=zeros(1,length(thval));
f12=@(tv) ((1/sqrt(2*pi*sigma_thsq))*exp(-(tv-mut).^2/(2*sigma_thsq)));
sct=integral(f12,at,bt,'ArrayValued',true);
pth(1)=integral(f12,at,thval(1)+(thval(2)-thval(1))/2,'ArrayValued',true)/sct;
for i=2:length(thval)-1
    pth(i)=integral(f12,thval(i)-(thval(i)-thval(i-1))/2,thval(i)+(thval(i+1)-thval(i))/2,'ArrayValued',true)/sct;
end
pth(length(thval))=integral(f12,thval(end)-(thval(end)-thval(end-1))/2,bt,'ArrayValued',true)/sct;

a=-5;
b=5;

% 
mux=0; % mean of source X
fx=@(xv) ((1/sqrt(2*pi*sigma_xsq))*exp(-(xv-mux).^2/(2*sigma_xsq)));  
scalx=integral(@(xv) fx(xv),a,b);
fx=@(xv) fx(xv)./scalx;

% non-strategic quantizer values
[xmall,ymall,Mv]=max_quant();

% quantizer values for encoders 1,2,3 for M=[2 4 8]
xthetaval1=zeros(length(thval),M+1,2);

for thn=1:length(thval)
% encoder 1: non-strategic
xthetaval1(thn,:,1)= [a xmall(find(Mv==M),1:M-1) b];

% encoder 2: dumb-strategic, Q_{NN}(x+theta), shifted quantizer 
% xthetaval1(thn,:,2)= [a+thval(thn) xmall(find(Mv==M),1:M-1) b+thval(thn)];
xthetaval1(thn,:,2)= [a-thval(thn) xmall(find(Mv==M),1:M-1)-thval(thn) b-thval(thn)];
end

m=find(M==Mv);
xnon=xmall(m,1:M-1);
xnon=[a xnon b];

xsamp=[a linspace(-3,3,numsamp(find(M==Mv))) b];

% fmincon conditions
A=[];
    b1=[];
    if M>2
        A=zeros((M-1-1)*length(thval),(M-1)*length(thval));
        A1=[1 -1 zeros(1,M-1-2+(M-1)*(length(thval)-1))];
        i=0;
        for j=1:length(thval)
        A(i+1,:)=A1;
        i1=i+1;
        for i=(j-1)*(M-1-1)+2:(j-1)*(M-1-1)+M-2
            A(i,:)=circshift(A(i-1,:),1);
        end
        if length(i)==0
            i=i1;
        end
        A1=circshift(A(i,:),2);
        end
        b1=zeros(size(A,1),1);
    end
    A;
    b1;


lb1=[a*ones(M-1,1)];
lb=repmat(lb1,length(thval),1);
ub1=[b*ones(M-1,1)];
ub=repmat(ub1,length(thval),1);


x0init=nchoosek(xsamp(2:end-1),M-1);
x0init=[a*ones(size(x0init,1),1) x0init b*ones(size(x0init,1),1)];
xrn1=randi(size(x0init,1),rn,length(thval));
xminit=zeros(rn,length(thval),M+1);
xminit(1,:,:)=repmat(xnon,length(thval),1);

for r=2:7
xminit(r,:,:)=[a*ones(length(thval),1) sort(-3+(6)*rand(length(thval),M-1)')' b*ones(length(thval),1)];
end
for r=8:rn
xminit(r,:,:)=x0init(xrn1(r,:)',:);
end
save(strcat('data',num2str(M),'.mat'),'xminit');



for lambda=lambdaval
p=[exp(-lambda) exp(-lambda)*lambda exp(-lambda)*lambda/2];
p=p./sum(p);
pdash=[exp(-lambda) exp(-lambda)*lambda];
pdash=pdash./sum(pdash);



for rho=rhoval
xrandinit=zeros(length(thval),M+1,rn); % all initializations
xrm=zeros(length(thval),M+1,rn); % final quantizer values for all initializations
erm=zeros(1,rn); % encoder distortions for all initializations
yrm=zeros(M,rn); % final quantizer representative values for all initializations
drm=zeros(1,rn); % decoder distortions for all initializations
exitflagrn=zeros(1,rn);

mux_corr=mux+rho*(sigma_xsq/sigma_thsq)^(1/2)*(thval(:)-mut); % mean of X conditional on theta 
sigma_xsq_corr=(1-rho^2)*sigma_xsq; % variance of X conditional on theta 
f1=@(xv,i) ((1/sqrt(2*pi*sigma_xsq_corr))*exp(-(xv-mux_corr(i)).^2/(2*sigma_xsq_corr)))*pth(i); % pdf of X conditional on theta

% non-revealing encoder distortion
xm=[a*ones(length(thval),1) b*ones(length(thval),1)];
ym=reconstruction(xm,thval,f1,p,xthetaval1);
dist_enc_nr=encoderdistortion_act(xm,ym,f1,thval,xthetaval1,p);
dist_dec_nr=decoderdistortion_act(xm,ym,f1,thval,p,xthetaval1);
[ym_enc2]=reconstruction_encoder2(xm,thval,f1,pdash,xthetaval1);
dist_enc2_nr=encoder2distortion(xm,ym_enc2,f1,thval);

% fully-revealing encoder distortion
xm=repmat(xnon,length(thval),1);
ym=reconstruction(xm,thval,f1,p,xthetaval1);
dist_enc_fr=encoderdistortion_act(xm,ym,f1,thval,xthetaval1,p);
dist_dec_fr=decoderdistortion_act(xm,ym,f1,thval,p,xthetaval1);
[ym_enc2]=reconstruction_encoder2(xm,thval,f1,pdash,xthetaval1);
dist_enc2_fr=encoder2distortion(xm,ym_enc2,f1,thval);

for r=1:rn
xrandinit(:,:,r)=xminit(r,:,:);
xm=reshape(xminit(r,:,:),length(thval),M+1);
x0=xm;
x0=sort(x0')';
x0=x0(:,2:end-1); % optimizing only decision values that are not boundaries
x0=x0';
x0=x0(:);
fun=@(x)f22fn(x,thval,f1,a,b,pdash,xthetaval1); % objective function 
% options = optimoptions('fmincon','MaxFunctionEvaluations',90000000,'MaxIterations',90000000,'Display','iter','PlotFcn',{@optimplotx,@optimplotfval,@optimplotfirstorderopt});
options = optimoptions('fmincon','MaxFunctionEvaluations',90000000,'MaxIterations',90000000,'SpecifyObjectiveGradient',true,'checkGradients',true,'Display','off');
tic
[x,fval,exitflag,output,lambda11,grad,hessian]=fmincon(fun,x0,A,b1,[],[],lb,ub,[],options); % gradient descent
toc
disp(strcat('Results for M=',num2str(M),', rho=',num2str(rho),',r=',num2str(r),',lambda=',num2str(lambda))); % display result for M

x=[a*ones(length(thval),1) reshape(x,M-1,length(thval))' b*ones(length(thval),1)] % gradient descent output
[ym_enc2]=reconstruction_encoder2(x,thval,f1,pdash,xthetaval1);
[ym_act]=reconstruction(x,thval,f1,p,xthetaval1);

exitflagrn(r)=exitflag
dist_enc2=encoder2distortion(x,ym_enc2,f1,thval);
dist_enc_act=encoderdistortion_act(x,ym_act,f1,thval,xthetaval1,p);
dist_dec_act=decoderdistortion_act(x,ym_act,f1,thval,p,xthetaval1);

xm=x;
xrm(:,:,r)=xm;
yrm_enc2(:,r)=ym_enc2;
yrm_act(:,r)=ym_act;

erm_enc2(r)=dist_enc2;
erm_act(r)=dist_enc_act;
drm_act(r)=dist_dec_act;

[in1 in2]=min(erm_enc2(1:r));
xm=xrm(:,:,in2);
[ym_enc2]=reconstruction_encoder2(x,thval,f1,pdash,xthetaval1);
[ym_act]=reconstruction(x,thval,f1,p,xthetaval1);
dist_enc2=encoder2distortion(xm,ym_enc2,f1,thval);
dist_enc_act=encoderdistortion_act(xm,ym_act,f1,thval,xthetaval1,p);
dist_dec_act=decoderdistortion_act(xm,ym_act,f1,thval,p,xthetaval1);

save(strcat('new2_threeenc_xtheta_lambda_',num2str(lambda),'M',num2str(M),'rho',num2str(rho),'varth',num2str(sigma_thsq),'varx',num2str(sigma_xsq),'noiseless_xtheta_gaussian.mat'),'xthetaval1','dist_enc2_nr','dist_enc2_fr','p','pdash','xm','ym_enc2','ym_act','dist_enc2','dist_enc_act','dist_dec_act','erm_enc2','erm_act','drm_act','xrm','xrandinit','exitflagrn','sigma_thsq','sigma_xsq','dist_enc_nr','dist_dec_nr','dist_enc_fr','dist_dec_fr');
end

[in1 in2]=min(erm_enc2);
xm=xrm(:,:,in2);
[ym_enc2]=reconstruction_encoder2(x,thval,f1,pdash,xthetaval1);
[ym_act]=reconstruction(x,thval,f1,p,xthetaval1);
dist_enc2=encoder2distortion(xm,ym_enc2,f1,thval);
dist_enc_act=encoderdistortion_act(xm,ym_act,f1,thval,xthetaval1,p);
dist_dec_act=decoderdistortion_act(xm,ym_act,f1,thval,p,xthetaval1);

save(strcat('new2_threeenc_xtheta_lambda_',num2str(lambda),'M',num2str(M),'rho',num2str(rho),'varth',num2str(sigma_thsq),'varx',num2str(sigma_xsq),'noiseless_xtheta_gaussian.mat'),'xthetaval1','dist_enc2_nr','dist_enc2_fr','p','pdash','xm','ym_enc2','ym_act','dist_enc2','dist_enc_act','dist_dec_act','erm_enc2','erm_act','drm_act','xrm','xrandinit','exitflagrn','sigma_thsq','sigma_xsq','dist_enc_nr','dist_dec_nr','dist_enc_fr','dist_dec_fr');
end
end
end


function [dist_dec]=decoderdistortion_act(xthetam,ym,f1,thval,pval,xthvalues)
M=size(xthetam,2)-1;
dist_dec=0;
for i=1:M
    for k=1:length(thval)
        f1temp= @(xv) f1(xv,k);
        f5=@(xv) (xv-ym(i))^2*f1temp(xv);
        dist_dec=dist_dec+pval(1)*integral(f5,xthvalues(k,i,1),xthvalues(k,i+1,1),'ArrayValued',true);
        dist_dec=dist_dec+pval(2)*integral(f5,xthvalues(k,i,2),xthvalues(k,i+1,2),'ArrayValued',true);
        dist_dec=dist_dec+pval(3)*integral(f5,xthetam(k,i),xthetam(k,i+1),'ArrayValued',true);
    end
end


function [dist_enc]=encoder2distortion(xthetam,ym,f1,thval)
M=size(xthetam,2)-1;
dist_enc=0;
for i=1:M
    for j=1:length(thval)
        f1temp= @(xv) f1(xv,j);
        f5=@(xv) (xv+thval(j)-ym(i))^2*f1temp(xv);
        dist_enc=dist_enc+integral(f5,xthetam(j,i),xthetam(j,i+1),'ArrayValued',true);
    end
end

 
function [dist_enc]=encoderdistortion_act(xthetam,ym,f1,thval,xthvalues,pval)
M=size(xthetam,2)-1;
dist_enc=0;
for i=1:M
for j=1:length(thval)
    f1temp= @(xv) f1(xv,j);
    f5=@(xv) (xv+thval(j)-ym(i))^2*f1temp(xv);
    for encind=1:2 
        dist_enc=dist_enc+pval(encind)*integral(f5,xthvalues(j,i,encind),xthvalues(j,i+1,encind),'ArrayValued',true);
    end
dist_enc=dist_enc+pval(3)*integral(f5,xthetam(j,i),xthetam(j,i+1),'ArrayValued',true);
end
end

function [ym]=reconstruction_encoder2(xthetam,thval,f1,pval,xthvalues)
M=size(xthetam,2)-1;
ym=zeros(1,M);
for i=1:M
num=0;
den=0;
for encind=1:2
for j=1:length(thval)
    f1temp= @(xv) f1(xv,j);
    f2=@(xv) xv*f1temp(xv);
    num=num+pval(encind)*integral(f2,xthvalues(j,i,encind),xthvalues(j,i+1,encind),'ArrayValued',true);
    den=den+pval(encind)*integral(f1temp,xthvalues(j,i,encind),xthvalues(j,i+1,encind),'ArrayValued',true);
end
end       
ym(i)=num/den;
end


function [ym]=reconstruction(xthetam,thval,f1,pval,xthvalues)
M=size(xthetam,2)-1;
ym=zeros(1,M);
for i=1:M
num=0;
den=0;
for encind=1:2
for j=1:length(thval)
    f1temp= @(xv) f1(xv,j);
    f2=@(xv) xv*f1temp(xv);
    num=num+pval(encind)*integral(f2,xthvalues(j,i,encind),xthvalues(j,i+1,encind),'ArrayValued',true);
    den=den+pval(encind)*integral(f1temp,xthvalues(j,i,encind),xthvalues(j,i+1,encind),'ArrayValued',true);
end
end       
for j=1:length(thval)
    f1temp= @(xv) f1(xv,j);
    f2=@(xv) xv*f1temp(xv);
    num=num+pval(3)*integral(f2,xthetam(j,i),xthetam(j,i+1),'ArrayValued',true);
    den=den+pval(3)*integral(f1temp,xthetam(j,i),xthetam(j,i+1),'ArrayValued',true);
end
ym(i)=num/den;
end


function [der]=derivative(xm,ym,f1,i,t,thval)
M=size(xm,2)-1;
der=0;
der=(xm(t,i)+thval(t)-ym(i-1))^2*f1(xm(t,i),t);
der=der-(xm(t,i)+thval(t)-ym(i))^2*f1(xm(t,i),t);
   
    
function [f22,der] = f22fn(x,thval,f1,a,b,pval,xthetaval1)
M=length(x)/length(thval)+1;
x=[a*ones(length(thval),1) reshape(x,M-1,length(thval))' b*ones(length(thval),1)];
[ym]=reconstruction_encoder2(x,thval,f1,pval,xthetaval1);
der=zeros(length(thval),M-1);
for i=2:M
    for t=1:length(thval)
    [der(t,i-1)]=derivative(x,ym,f1,i,t,thval);
    end
end
der=der';
der=der(:);
f22=encoder2distortion(x,ym,f1,thval);


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
