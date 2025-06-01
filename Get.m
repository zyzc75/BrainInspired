SubjectNums = 23;
testNums = 18;
channelNums = 14;
rhythms = cell(1,SubjectNums);
for s = 1 : SubjectNums
    dtw = cell(channelNums,testNums);
    for t = 1 : testNums
        for c = 1 : channelNums
            stemp = data{t,s}(1:3840,c);
            dtw{c,t} = getRhythm(stemp');
        end
    end
    rhythms{1,s} = dtw;
end

function rhythms = getRhythm(inputSignal)
    [c,l]=wavedec(inputSignal,4,'db4');
    a5=wrcoef('a',c,l,'db4',4); % 0-4
    d5=wrcoef('d',c,l,'db4',4); % 4-8
    d4=wrcoef('d',c,l,'db4',3); % 8-16
    d3=wrcoef('d',c,l,'db4',2); % 16-32
    d2=wrcoef('d',c,l,'db4',1); % 32-64

    rhythms = [d2;d3;d4;d5;a5];
end

% for s = 1 : 23
%     R = rhythms{1,s};
%     for r = 1 : 5
%         Ftemp = zeros(14*18,3840);
%         Ltemp = zeros(14*18,3);
%         for c = 1 : 14
%             for t = 1 : 18
%                 Ftemp((c-1)*18+t,:) = R{c,t}(r,:);
%                 Ltemp((c-1)*18+t,1) = ALable(t,s);
%                 Ltemp((c-1)*18+t,2) = VLable(t,s);
%                 Ltemp((c-1)*18+t,3) = DLable(t,s);
%             end
%         end
%         FEATURES{r,s} = Ftemp;
%         LABELS{r,s} = Ltemp;
%     end
% end
% 
% for s = 1 : 23
%     for r = 1 : 5 
%         Ltemp = LABELS{r,s};
%         Ltemp = double(Ltemp>=3);
%         LABELS{r,s} = Ltemp;
%     end
% end