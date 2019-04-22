function [ECGpeaks] = QSpeaks( ECG,Rposition,fs )
%Q,S peaks detection
% point to point time duration is determined by sampling frequency: 1/fs
% the duration of QRS complex varies from 0.1s to 0.2s 
% complex--fs*0.2
aveHB = length(ECG)/length(Rposition);
fid_pks = zeros(length(Rposition),7);
% P QRSon Q R S QRSoff 
%% set up searching windows
windowS = round(fs*0.1); windowQ = round(fs*0.05);
windowP = round(aveHB/3); windowT = round(aveHB*2/3);
windowOF = round(fs*0.04);
% initialization
for i = 1:length(Rposition)
    thisR = Rposition(i);
    if i==1
        fid_pks(i,4) = thisR;
        fid_pks(i,6) = thisR+windowS;
    elseif i==length(Rposition)
        fid_pks(i,4) = thisR;
        %(thisR+windowT) < length(ECG) && (thisR - windowP) >=1
        fid_pks(i,2) = thisR-windowQ;
    else
        
        if (thisR+windowT) < length(ECG) && (thisR - windowP) >=1
        % Q S peaks 
        fid_pks(i,4) = thisR;
        [Sv,Sp] = min(ECG(thisR:thisR+windowS));
        thisS = Sp + thisR-1;
        fid_pks(i,5) = thisS;
        [Qv,Qp] = min(ECG(thisR-windowQ:thisR));
        thisQ = thisR-(windowQ+1) + Qp;
        fid_pks(i,3)=thisQ;      
        % onset and offset detection
        interval_q = ECG(thisQ-windowOF:thisQ);
        [ ind ] = onoffset(interval_q,'on' );
        thisON = thisQ - (windowOF+1) + ind;
        interval_s = ECG(thisS:thisS+windowOF);
        [ ind ] = onoffset( interval_s,'off' );
        thisOFF = thisS + ind-1;
        fid_pks(i,2) = thisON;
        fid_pks(i,6) = thisOFF;       
%         % T and P waves detection
%         lastOFF = fid_pks(i-1,6);
%         nextON = 
        end
    end       
end
%%
% P,T detection
%   Detection T waves first and distinguish the type of T waves
for i = 2:length(Rposition)-1
    lastOFF = fid_pks(i-1,6);
    thisON = fid_pks(i,2);
    thisOFF = fid_pks(i,6);
    nextON = fid_pks(i+1,2);
    if thisON>lastOFF && thisOFF<nextON
       Tzone = (thisOFF:(nextON-round((nextON-thisOFF)/3)));
       Pzone = ((lastOFF+round(2*(thisON-lastOFF)/3)):thisON);
       [Tpks,thisT] = max(ECG(Tzone));
       [Ppks,thisP] = max(ECG(Pzone));
       fid_pks(i,1) = (lastOFF+round(2*(thisON-lastOFF)/3)) + thisP -1;
       fid_pks(i,7) = thisOFF + thisT -1;
    end
end
ECGpeaks = [];
Ind = zeros(length(Rposition),1);
for i = 1:length(Rposition)
    Ind(i) = prod(fid_pks(i,:));
    if Ind(i) ~= 0
        ECGpeaks = [ECGpeaks;fid_pks(i,:)];
    end
end


end  
% Tpks = [];
% Tposition =[];
% Ttype = [];
% Ppks = [];
% Pposition =[];
% if length(QRS_OFF) <= length(QRS_ON)
%     for i = 1:length(QRS_OFF)-1
%         lengthTzone = int64((QRS_ON(i+1)-QRS_OFF(i))*2/3);
%         if lengthTzone <= 0 
%             lengthTzone = abs(lengthTzone);
%             disp('abnormal heart beat');
%         end
%         if QRS_OFF(i)+lengthTzone-1 > length(ECG)
%             Tzone = ECG(QRS_OFF(i):length(ECG));
%         else
%             Tzone = ECG(QRS_OFF(i):QRS_OFF(i)+lengthTzone-1);
%         end
% %         if length(max(Tzone))~=1
% %             deb = 1;
% %         end
%         [Tpks(end+1),Tind] = max(Tzone);
%         Tposition(end+1) = QRS_OFF(i)+Tind-1;
%         lengthPzone = int64((-QRS_OFF(i)+QRS_ON(i+1))/3);
%         if lengthPzone <= 0 
%             lengthPzone = abs(lengthPzone);
%             disp('abnormal heart beat');
%         end
%         Pzone = ECG(QRS_ON(i+1)-lengthPzone-1:QRS_ON(i+1));
%         [Ppks(end+1),Pind] = max(Pzone);
%         Pposition(end+1) = QRS_ON(i+1)+Pind-lengthPzone;
%     end
% else
%     for i = 1:length(QRS_ON)-1
%         lengthTzone = int64((QRS_ON(i+1)-QRS_OFF(i))*2/3);
%         if QRS_OFF(i)+lengthTzone-1 > length(ECG)
%             Tzone = ECG(QRS_OFF(i):length(ECG));
%         else
%             Tzone = ECG(QRS_OFF(i):QRS_OFF(i)+lengthTzone-1);
%         end
%         [Tpks(end+1),Tind] = max(abs(Tzone));
%         Tposition(end+1) = QRS_OFF(i)+Tind-1;
%         lengthPzone = int64((-QRS_OFF(i)+QRS_ON(i+1))/3);
%         Pzone = ECG(QRS_ON(i+1)-lengthPzone-1:QRS_ON(i+1));
%         [Ppks(end+1),Pind] = max(Pzone);
%         Pposition(end+1) = QRS_ON(i+1)+Pind-lengthPzone;
%     end
% end
% for i = 1:length(Tpks)
%     if Tpks(i)>0
%         display('T Upright');
%     else
%         display('T Inverted');
%     end
% end
% hold on;plot(Tposition,Tpks,'o');
% hold on;plot(Pposition,Ppks,'*');
 %%   
% [pks,locs] = findpeaks(A5);hold on; plot(locs,pks,'*');
% 
% for i = 1:length(S)
%     SearchArea = [];
%     for j = 1:length(locs)
%         if locs(j) > S(i)
%             SearchArea(end+1) = locs(j);
%         end
%     end
%     su = [];
%     for k = 1:length(SearchArea)
%         su(end+1)  = SearchArea(k)-S(i);
%     end
%     if ~isempty(su)
%         Tpks(end+1) = S(i) + min(su);
%     end
% end
% 
% % P detection
% 
% 
% for i = 1:length(Q)
%     SearchArea = [];
%     for j = 1:length(locs)
%         if locs(j) < Q(i)
%             SearchArea(end+1) = locs(j);
%         end
%     end
%     su = [];
%     for k = 1:length(SearchArea)
%         su(end+1)  = Q(i)-SearchArea(k);
%     end
%     if ~isempty(su)
%         Ppks(end+1) = Q(i) - min(su);
%     end
% end
% % check the result
% for i = 1:length(Tpks)
%     T(i) = A5(Tpks(i));
% end
% for i = 1:length(Ppks)
%     P(i) = A5(Ppks(i));
% end
% figure(5);plot(A5);
% hold on;plot(Tpks,T,'*');
% hold on;plot(Ppks,P,'+');

%%
%Oops after interpolation, we slightly deviate from the true value
%Thus we need to search for the local minimum in the whole signal again
% Qposition = int64(Q.*Scale);
% Sposition = int64(S.*Scale);
% 
% for i = 1:length(R)
%     [Sv,Sp] = min(ECG(Sposition(i):Sposition(i)+20));
%     Sposition(i) = Sp + Sposition(i)-1;
%     [Qv,Qp] = min(ECG(Qposition(i)-20:Qposition(i)));
%     Qposition(i) = Qposition(i)-21 + Qp;
% end
% 
% Pposition = int64(Ppks.*Scale);
% Tposition = int64(Tpks.*Scale);
% 
% for i = length(Pposition)
%     [Pv,Pp] = max(ECG(Pposition(i)-20:Pposition(i)));
%     Pposition(i) = Pposition(i)-21 + Pp;
% end
% 
% for i = length(Tposition)
%     [Tv,Tp] = max(ECG(Tposition(i):Tposition(i)+20));
%     Tposition(i) = Tp + Tposition(i)-1;
% end
% ecgS = [];
% for i = 1:2271
%     ecgS(end+1) = issorted(ECGpeaks(1,:));
% end
% prod(ecgS)