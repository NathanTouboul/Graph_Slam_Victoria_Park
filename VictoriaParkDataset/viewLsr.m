function ViewLsr

%function ViewLsr(FileName,figu,dttt)
% Jose. ACFR. 1999.

FileName = 'aa3_lsr2.mat' ;
dttt = 0.2 ;
figu = 1 ;

global AAr; 
AAr = [0:360]*pi/360 ;
load(FileName) ;
L = size(LASER) ; L=L(1) ;
Time = double(TLsr) ; clear TLsr; 
CAAA = cos(AAr) ;
SAAA = sin(AAr) ;

global pCircles ;
nc = 9 ; aaa = [0:nc]*2*pi/nc ; pCircles = [ cos(aaa); sin(aaa) ] ;  


figure(figu) ;clf ; 
zoom on ;
hhh =plot(0,0,'.','erasemode','xor') ;   %laser
hold on;
hhh2=plot(0,0,'o','erasemode','xor') ;  % landmarks centers
hhh3=plot(0,0,'g','erasemode','xor') ;   % approx. landm. circles
axis([-50,50,0,75]);
hold off ;

%Mask13 = uint16(2^13 -1) ;
%MaskA  = bitcmp(Mask13,16) ;

Mask13 = bitshift(2^13-1,1);
MaskA  = bitand(Mask13,16);

lidar_ranges = zeros(1);
lidar_bearings = zeros(1);
lidar_signatures = zeros(1);

for i=1:L-1
    
	RR = double(  bitand( Mask13,LASER(i,:)) ) ;
	a  = uint16(  bitand( MaskA ,LASER(i,:)) ) ;
	ii = find(a>0) ;
	RR = RR/100 ;
	xra=detectTreesI16(RR) ;
    
    if length(xra) < 3
        continue
    end

	ii2 = find(RR<75) ;
	xx = RR(ii2).*CAAA(ii2) ;
	yy = RR(ii2).*SAAA(ii2) ;
  	
    %set(hhh,'XData',xx,'YData',yy) ;	
	
    xl = xra(1,:).*cos(xra(2,:)) ;
	yl = xra(1,:).*sin(xra(2,:)) ;
  	
    %set(hhh2,'XData',xl,'YData',yl) ;	
	
    %DibuTrees(xl,yl,xra(3,:),hhh3) ;

    % Saving range, bearing, signature
    nb_cols = size(xra, 2);
    lidar_ranges(i, 1:nb_cols) = xra(1, :);
    lidar_bearings(i, 1:nb_cols) = xra(2, :);
    lidar_signatures(i, 1:nb_cols) = xra(3, :);
    
	pause(dttt) ;
   


save('lidar_measurements.mat', 'lidar_ranges', 'lidar_bearings', 'lidar_signatures')
end;
return ;

% --------------------------------------------


function DibuTrees(xl,yl,rl,hdl)
global pCircles ;
xyCi = pCircles ;
nc = size(xyCi) ; nc = nc(2) ;
nl = length(xl) ;
u=1 ;
z=1 ;
xyAllC = zeros(2,nc*nl+nl) ;
xxx    = zeros(2,nl) ;

for i=1:nl
    
	Ri=rl(i)*0.5 ;
	xyCi(1,:)= pCircles(1,:)*Ri + xl(i) ;
	xyCi(2,:)= pCircles(2,:)*Ri + yl(i) ;
	xyAllC(:,u:u+nc-1)=xyCi ;
	xxx(:,nl-i+1) = xyCi(:,1) ;
	u=u+nc ;
end 
xyAllC(:,u:end) = xxx ;
set(hdl,'XData',xyAllC(1,:),'YData',xyAllC(2,:)) ;	
return ;


% --------------------------------------------
	
