RESULT2 = cell(5,5);
for r = 1 : 5
    parfor s = 1 : 23
        result21(s,:) = NFoldTest2(FEATURES{r,s},LABELS{1,s}(:,1),10);
        result22(s,:) = NFoldTest2(FEATURES{r,s},LABELS{1,s}(:,2),10);
        result23(s,:) = NFoldTest2(FEATURES{r,s},LABELS{1,s}(:,3),10);
    end
    RESULT2{r,1} = result21;
    RESULT2{r,2} = result22;
    RESULT2{r,3} = result23;
end

% RESULT3 = cell(5,5);
% for r = 1 : 5
%     parfor s = 1 : 18
%         result1(s,:) = NFoldTest3(FEATURES{r,s},LABELS{1,s}(:,1),10);
%         result2(s,:) = NFoldTest3(FEATURES{r,s},LABELS{1,s}(:,2),10);
%         result3(s,:) = NFoldTest3(FEATURES{r,s},LABELS{1,s}(:,3),10);
%     end
%     RESULT3{r,1} = result1;
%     RESULT3{r,2} = result2;
%     RESULT3{r,3} = result3;
%     RESULT3{r,4} = result4;
%     RESULT3{r,5} = result5;
% end

RESULT4 = cell(5,5);
for r = 1 : 5
    parfor s = 1 : 23
        result41(s,:) = NFoldTest4(FEATURES{r,s},LABELS{1,s}(:,1:2),10);
    end
    RESULT4{r,1} = result41;
end