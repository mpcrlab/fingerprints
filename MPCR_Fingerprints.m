%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------%
%
% Machine Perception and Cognitive Robotics Laboratory
%
%     Center for Complex Systems and Brain Sciences
%
%              Florida Atlantic University
%
%------------------------------------------------------%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%------------------------------------------------------%
% Data:
% http://bias.csr.unibo.it/fvc2000/
%------------------------------------------------------%
function MPCR_Fingerprints
clear all
close all
clc

N=10; %Number of fingers
M=8;  %Number of photos per finger

lambda=0.1;

[L,finger_names,finger_names_key]=load_library(N,M);

r=randperm(size(L,2));

L=L(:,r);
finger_names_key=finger_names_key(r);

% for i=1:size(L,2)
%
%     imagesc(reshape(L(:,i),120,128))
%     colormap(gray)
%     finger_names_key(i)
%     finger_names{finger_names_key(i)}
%     pause
%
% end


L1=L;

p1=[];

t=[];

for k=1:size(L,2)%randperm(size(L,2))%
    
    L=L1;
    y=L(:,k);
    L(:,k)=0;
    
    a=LCA(y, L, lambda);
   
    b=[];
    
    for j=1:N 
        b=[b sum(abs(a(find(finger_names_key==j))))];
    end
    
    [b1,b2]=max(b);
    
    figure(1)
    subplot(511)
    bar(b)
    
    p=[b2 finger_names_key(k)]
    p1=[p1; p];
     
    subplot(512)
    plot(p1)
            
    subplot(513)
    imagesc(reshape(y,478,448))
    colormap(gray)
    
    subplot(514)
    bar(abs(a))
    
    subplot(515)
    t=[t b2==finger_names_key(k)];
    hist(t,2);
    sum(t)/length(t)
     
    drawnow()
   
    
end


end




function [L,fingerprint_names,finger_names_key]=load_library(N,M)

cd('DB3_B')

% fingerprint_names={'Fingerprint_101','Fingerprint_102','Fingerprint_103','Fingerprint_104','Fingerprint_105','Fingerprint_106','Fingerprint_107','Fingerprint_108','Fingerprint_109','Fingerprint_110'};
fingerprint_names={'101','102','103','104','105','106','107','108','109','110'};
finger_names_key=[];%ceil((1:(M*N))/M);

L=[]; %Library

for i =1:size(fingerprint_names,2)
    
    fingerprint_names{i};
    
    dr1=dir([fingerprint_names{i} '*.tif']);
    
    f1={dr1.name}; % get only filenames to cell
    
    D=[]; %Dictionary
    
    for j=1:M%length(f1) % for each image
        
        a1=f1{j};  
        b1=im2double(imread(a1));
        b1=b1(1:end)';
        b1 = b1 - min(b1(:));
        b1 = b1 / max(b1(:));
       
        D=[D b1];
        finger_names_key=[finger_names_key i];
        
    end
    
%   D = bsxfun(@minus,D,mean(D)); %remove mean
    fX = fft(fft(D,[],2),[],3); %fourier transform of the images
    spectr = sqrt(mean(abs(fX).^2)); %Mean spectrum
    D = ifft(ifft(bsxfun(@times,fX,1./spectr),[],2),[],3); %whitened L
    
    L=[L D];
    
    
end

% L = bsxfun(@minus,L,mean(L)); %remove mean
fX = fft(fft(L,[],2),[],3); %fourier transform of the images
spectr = sqrt(mean(abs(fX).^2)); %Mean spectrum
L = ifft(ifft(bsxfun(@times,fX,1./spectr),[],2),[],3); %whitened L


end



function [a, u] = LCA(y, D, lambda)

t=.01;
h=.00000001;

d = h/t;
u = zeros(size(D,2),1);

for i=1:1000
    
    a=u.*(abs(u) > lambda);
    u =   u + d * ( D' * ( y - D*a ) - u - a  ) ;

end


end



